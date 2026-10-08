"""PROTOTYPE. Tracks the tasks of external consumers that consume the executor's
outputs.

An external consumer takes outputs from the executor and consumes them in tasks
the executor doesn't run, such as the push-based split's delivery of a block to
a train worker. It reports each of those tasks here from its own threads. The
executor drains the reports once per scheduling step, on its own thread, the
only thread that touches the lineage tracker.

Each external task is registered with the lineage tracker as a child of the
tasks that produced its blocks. If its input was lost, it is the target of a
reconstruction plan like a task inside the executor that hit a lost object. Only the blocks
it consumed are re-produced. The re-produced blocks are stamped for the external
task and leave the executor like any other output, so the consumer submits a
new external task for them.

TODO(push-split): garbage collect the lineage that completed external tasks
make unreachable. A task's lineage is needed while any of its downstream outputs
can still be lost, so collect bottom-up. A completed or abandoned external task
is collectable. Any other task is collectable once it completed, no
reconstruction plan is in flight through it, and every output it queued has a
collected consumer. Collecting a task can make its parents collectable.
Collecting a seed task releases its retained input
(``MapOperator._seed_task_inputs``). Without external consumers the final
operator's outputs never get a consumer, so nothing is collected and the default
path is unchanged.
"""

import itertools
import threading
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Set, Tuple

from ray.data._internal.execution.interfaces import RefBundle
from ray.data._internal.execution.interfaces.ref_bundle import ReconstructionStamp
from ray.exceptions import ObjectLostError
from ray.types import ObjectRef

if TYPE_CHECKING:
    from ray.data._internal.execution.lineage_tracker import LineageTracker

# Identifies one external task. Returned by ``ExternalConsumerTasks.submit`` and
# passed back to report the task's outcome.
ExternalTaskHandle = int


class ExternalTaskLostInput(NamedTuple):
    """An external task whose input was lost, to be reconstructed by the
    executor."""

    lineage_task_id: str
    reconstruction_plan_id: Optional[str]
    error: ObjectLostError


class ExternalConsumerTasks:
    """Records the external tasks that consumer threads report, and applies them
    to the lineage tracker on the executor thread.

    Every method is thread-safe except ``update_lineage`` and
    ``pop_lost_inputs``, which only the executor thread calls, in that order.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._next_handle = itertools.count()
        # Blocks the executor handed out that no external task was submitted
        # for yet, with the stamp of the bundle they came in. A stamped bundle
        # is a re-produced input of a failed external task.
        self._outputs_awaiting_submission: Dict[
            ObjectRef, Optional[ReconstructionStamp]
        ] = {}
        # Reports not yet registered. A task is always submitted before it can
        # complete, lose its input, or be abandoned, so registering every submission of a
        # drain before any outcome keeps them in order.
        self._submitted: List[
            Tuple[ExternalTaskHandle, List[ObjectRef], Optional[ReconstructionStamp]]
        ] = []
        self._completed: List[ExternalTaskHandle] = []
        self._lost_inputs: List[Tuple[ExternalTaskHandle, ObjectLostError]] = []
        self._abandoned: List[ExternalTaskHandle] = []
        # Tasks submitted but not yet completed, lost, or abandoned.
        self._num_unresolved = 0
        # Output splits whose consumer stopped early.
        self._finished_splits: Set[int] = set()

        # Executor thread only. The lineage task and reconstruction plan each
        # registered external task runs as.
        self._lineage_ids: Dict[ExternalTaskHandle, Tuple[str, Optional[str]]] = {}
        # Executor thread only. Lost inputs set aside by `update_lineage` for
        # `pop_lost_inputs`.
        self._lost_inputs_to_pop: List[ExternalTaskLostInput] = []

    def on_output_taken(self, bundle: RefBundle) -> None:
        """Called by the executor when an external consumer takes an output.

        Keeps the executor alive until an external task is submitted for each
        of the output's blocks, and remembers the stamp of a re-produced block.
        """
        stamp = bundle.reconstruction_stamp
        assert stamp is None or len(bundle.blocks) == 1, bundle
        with self._lock:
            for entry in bundle.blocks:
                self._outputs_awaiting_submission[entry.ref] = stamp

    def submit(self, block_refs: List[ObjectRef]) -> ExternalTaskHandle:
        """Submit an external task that consumes blocks the executor handed out.

        Args:
            block_refs: The blocks the task consumes. Each must have been handed
                out by the executor, and not consumed by another task yet.

        Returns:
            A handle to report the task's outcome with.
        """
        with self._lock:
            # A re-produced block arrives alone, so at most one stamp applies.
            stamps = {self._outputs_awaiting_submission.pop(ref) for ref in block_refs}
            stamps.discard(None)
            assert len(stamps) <= 1, stamps
            stamp = stamps.pop() if stamps else None
            handle = next(self._next_handle)
            self._submitted.append((handle, list(block_refs), stamp))
            self._num_unresolved += 1
            return handle

    def complete(self, handle: ExternalTaskHandle) -> None:
        """The external task received its input."""
        with self._lock:
            self._completed.append(handle)
            self._num_unresolved -= 1

    def report_lost_input(
        self, handle: ExternalTaskHandle, error: ObjectLostError
    ) -> None:
        """The external task's input was lost before it was received. The
        executor re-produces it and hands it out again, possibly to a different
        split.

        Only for a lost input. Abandon a task that failed for any other reason,
        since reconstruction can't help it.
        """
        with self._lock:
            self._lost_inputs.append((handle, error))
            self._num_unresolved -= 1

    def abandon(self, handle: ExternalTaskHandle) -> None:
        """The external task ended without its input for a reason other than
        data loss, e.g. its consumer died. The input isn't re-produced."""
        with self._lock:
            self._abandoned.append(handle)
            self._num_unresolved -= 1

    def finish_split(self, output_split_idx: int) -> None:
        """The consumer of ``output_split_idx`` stopped. Its queued outputs no
        longer keep the executor running.

        A re-produced block that the output splitter routes to a finished split
        is dropped. That's acceptable, since a split that stops early already
        makes the epoch partial.
        """
        with self._lock:
            self._finished_splits.add(output_split_idx)

    def finished_splits(self) -> Set[int]:
        with self._lock:
            return set(self._finished_splits)

    def has_unresolved(self) -> bool:
        """Whether any taken output or submitted task isn't resolved yet,
        including reports the executor hasn't registered."""
        with self._lock:
            return bool(
                self._outputs_awaiting_submission
                or self._num_unresolved
                or self._completed
                or self._lost_inputs
                or self._abandoned
            )

    def update_lineage(self, lineage_tracker: Optional["LineageTracker"]) -> None:
        """Update the lineage external consumer task status since the last call.
        Executor thread only.

        Registers submitted tasks and completes completed ones with
        ``lineage_tracker``, if there is one. Drops abandoned ones. Sets the
        lost inputs reported with them aside for ``pop_lost_inputs``. Their
        tasks are registered by then, since a task is always submitted before
        its input can be reported lost.

        Args:
            lineage_tracker: The executor's lineage tracker, or None if lineage
                reconstruction is disabled.
        """
        with self._lock:
            submitted, self._submitted = self._submitted, []
            completed, self._completed = self._completed, []
            lost_inputs, self._lost_inputs = self._lost_inputs, []
            abandoned, self._abandoned = self._abandoned, []

        for handle, block_refs, stamp in submitted:
            if stamp is not None:
                lineage_ids = (stamp.lineage_task_id, stamp.reconstruction_plan_id)
            else:
                lineage_ids = (f"consumer:{handle}", None)
            self._lineage_ids[handle] = lineage_ids
            if lineage_tracker is not None:
                dependencies = lineage_tracker.resolve_dependencies(
                    [ref.hex() for ref in block_refs]
                )
                lineage_tracker.register_task_submission(
                    lineage_ids[0], dependencies, lineage_ids[1]
                )

        for handle in completed:
            lineage_task_id, plan_id = self._lineage_ids.pop(handle)
            if lineage_tracker is not None:
                lineage_tracker.register_task_complete(lineage_task_id, plan_id)

        for handle in abandoned:
            del self._lineage_ids[handle]

        self._lost_inputs_to_pop.extend(
            ExternalTaskLostInput(*self._lineage_ids.pop(handle), error)
            for handle, error in lost_inputs
        )

    def pop_lost_inputs(self) -> List[ExternalTaskLostInput]:
        """Return the lost inputs set aside by ``update_lineage``, and forget
        them. Executor thread only.

        A loss reported after the last ``update_lineage`` isn't returned until
        the next one registers its task.

        Returns:
            The tasks whose input was lost, for the executor to reconstruct.
        """
        lost_inputs, self._lost_inputs_to_pop = self._lost_inputs_to_pop, []
        return lost_inputs
