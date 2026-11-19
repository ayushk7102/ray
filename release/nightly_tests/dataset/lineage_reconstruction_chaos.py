"""Raylet chaos and lineage reconstruction switches shared by the dataset benchmarks.

``add_chaos_args`` adds the flags. ``ray_init_kwargs`` turns ``--lineage-reconstruction``
into the job config that hands lost objects to Ray Data instead of Ray Core.
``start_chaos`` and ``stop_chaos`` run the raylet killer, and
``install_lineage_reconstruction_counter`` counts how often Ray Data reconstructed.
"""

import argparse
import asyncio
import random
import time
from typing import Any, Dict

import ray
from ray._common.network_utils import build_address
from ray._private.test_utils import NodeKillerBase, _shutdown_raylet
from ray.job_config import JobConfig
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# Chaos starts once the pipeline is about to run, then waits this long so a
# backlog of produced blocks exists to lose.
CHAOS_KILL_DELAY_S = 60


def add_chaos_args(parser: argparse.ArgumentParser, min_alive_workers: int) -> None:
    """Add ``--chaos``, its knobs and ``--lineage-reconstruction`` to ``parser``."""
    parser.add_argument(
        "--chaos",
        action="store_true",
        help=(
            "Whether to enable chaos. If set, this script kills worker raylets, so "
            "the objects on those nodes are lost."
        ),
    )
    parser.add_argument(
        "--chaos-kill-interval-s",
        type=float,
        default=60,
        help="Seconds between kill rounds. Each round lands at a random point in it.",
    )
    parser.add_argument(
        "--chaos-kill-batch-size",
        type=int,
        default=2,
        help=(
            "Worker raylets killed together in each round. Above 1, several "
            "lineage reconstructions run at once."
        ),
    )
    parser.add_argument(
        "--chaos-max-kills",
        type=int,
        default=None,
        help="Stop after this many raylets are killed. Unlimited by default.",
    )
    parser.add_argument(
        "--chaos-min-alive-workers",
        type=int,
        default=min_alive_workers,
        help=(
            "Never kill a raylet if fewer than this many worker nodes would stay "
            "alive. Chaos pauses until replacement nodes join."
        ),
    )
    parser.add_argument(
        "--lineage-reconstruction",
        action="store_true",
        help=(
            "Start the driver with "
            "JobConfig(_disable_job_level_lineage_reconstruction=True), so Ray Data "
            "lineage reconstruction rebuilds lost objects instead of Ray Core. With "
            "--chaos, fail the run if lineage reconstruction never ran."
        ),
    )


def ray_init_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """The ``ray.init`` kwargs for ``--lineage-reconstruction``."""
    if not args.lineage_reconstruction:
        return {}
    # Ray Data lineage reconstruction is gated on this job config. It also turns Ray
    # Core lineage reconstruction off for the job.
    return {"job_config": JobConfig(_disable_job_level_lineage_reconstruction=True)}


def install_lineage_reconstruction_counter() -> Dict[str, int]:
    """Count Ray Data lineage reconstructions for this run.

    Wraps ``LineageTracker.register_task_failed``, which the executor calls once
    per detected loss, so the run can assert lineage reconstruction actually ran
    rather than passing because chaos never lost a needed object. The streaming
    executor runs in the driver process, so patching the class here observes its
    tracker.

    Counts only calls that return seed ids. An empty list means reconstruction of
    that lineage was already under way and nothing was resubmitted.
    """
    from ray.data._internal.execution.lineage_tracker import LineageTracker

    assert ray.data.DataContext.get_current().enable_ray_data_reconstruction, (
        "--lineage-reconstruction is set, but Ray Data lineage reconstruction is "
        "off. Check that the driver started with "
        "JobConfig(_disable_job_level_lineage_reconstruction=True)."
    )
    state = {"reconstructions": 0}
    original = LineageTracker.register_task_failed

    def counting_register_task_failed(self, *args, **kwargs):
        seed_task_ids, plan_id = original(self, *args, **kwargs)
        if seed_task_ids:
            state["reconstructions"] += 1
        return seed_task_ids, plan_id

    LineageTracker.register_task_failed = counting_register_task_failed
    return state


def assert_lineage_reconstruction_ran(state: Dict[str, int]) -> None:
    """Fail a chaos run in which Ray Data lineage reconstruction never ran.

    The run completing proves the dataset survived chaos. This asserts it survived
    *because of* lineage reconstruction. Without it a run where chaos never lost a
    needed object passes silently and verifies nothing about reconstruction.
    """
    assert state["reconstructions"] > 0, (
        "Chaos was enabled but Ray Data lineage reconstruction never ran, so "
        "this run did not exercise lineage reconstruction."
    )


@ray.remote(num_cpus=0)
class FlooredRayletKiller(NodeKillerBase):
    """Kills worker raylets, but keeps at least `min_alive_workers` workers alive.

    Killing the raylet takes the node's object store down with it, so the
    objects it held are lost at once. A terminator with a grace period drains the
    node first, so nothing is lost and lineage reconstruction never runs.

    `NodeKillerBase` only keeps one worker alive, so with unlimited kills a cluster
    that replaces nodes slowly would shrink to a single node. The floor bounds how
    much capacity chaos can take at once. Kills resume as replacement nodes join.
    """

    def __init__(self, *args, min_alive_workers: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_alive_workers = min_alive_workers

    async def _find_resources_to_kill(self):
        while self.is_running:
            # A killed node can still read as alive until the GCS notices, so the
            # killed set is excluded as well.
            alive_workers = [
                (n["NodeID"], n["NodeManagerAddress"], n["NodeManagerPort"])
                for n in ray.nodes()
                if n["Alive"] and n["NodeID"] != self.head_node_id
            ]
            alive_workers = [w for w in alive_workers if w not in self.killed]
            if len(alive_workers) - self.batch_size_to_kill >= self._min_alive_workers:
                return random.sample(alive_workers, self.batch_size_to_kill)
            await asyncio.sleep(5)
        return []

    async def _kill_resource(self, node_id: str, ip: str, port: int) -> bool:
        if port is None:
            return False
        _shutdown_raylet(build_address(ip, port), graceful=False)
        return True

    def num_killed(self) -> int:
        return len(self.killed)


def stop_chaos(resource_killer, timeout_s: float = 300) -> None:
    """Stop killing raylets, then wait until the GCS marks every killed node dead.

    The GCS takes a while to notice a killed raylet. Until it does, calls that fan
    out to every alive raylet fail on that node.
    """
    ray.get(resource_killer.stop_run.remote())
    # Waits for the kill loop to exit, so a kill in flight during `stop_run` is
    # included.
    killed = {
        node_id for node_id, _, _ in ray.get(resource_killer.get_killed_nodes.remote())
    }
    deadline = time.monotonic() + timeout_s
    while True:
        still_alive = {n["NodeID"] for n in ray.nodes() if n["Alive"]} & killed
        if not still_alive:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"{len(still_alive)} killed nodes still read as alive after "
                f"{timeout_s}s: {sorted(still_alive)}"
            )
        time.sleep(5)


def start_chaos(args: argparse.Namespace):
    assert ray.is_initialized()

    head_node_id = ray.get_runtime_context().get_node_id()
    scheduling_strategy = NodeAffinitySchedulingStrategy(
        node_id=head_node_id, soft=False
    )
    resource_killer = FlooredRayletKiller.options(
        scheduling_strategy=scheduling_strategy
    ).remote(
        head_node_id,
        kill_interval_s=args.chaos_kill_interval_s,
        kill_delay_s=CHAOS_KILL_DELAY_S,
        max_to_kill=args.chaos_max_kills,
        batch_size_to_kill=args.chaos_kill_batch_size,
        min_alive_workers=args.chaos_min_alive_workers,
    )

    ray.get(resource_killer.ready.remote())

    resource_killer.run.remote()
    return resource_killer
