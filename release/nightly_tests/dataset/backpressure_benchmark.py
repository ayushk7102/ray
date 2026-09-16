import argparse
import functools
import time

import numpy as np
import pyarrow as pa
import ray
from ray._private.test_utils import (
    EC2InstanceTerminator,
    EC2InstanceTerminatorWithGracePeriod,
    RayletKiller,
    WorkerKillerActor,
)
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from benchmark import Benchmark


# Allow 20% headroom over the observed 10.8 GiB baseline.
# Chaos killers selectable via --chaos-type. TerminateEC2Instance (the default)
# terminates the instance with no grace period, so its in-store objects are lost
# immediately -- the failure mode that exercises Ray Data lineage recovery.
# KillWorker kills the worker process running a task (the block's producer), so
# the objects it just produced are lost while the node stays up -- the most
# targeted "kill an object owner" failure mode.
CHAOS_KILLERS = {
    "KillRaylet": RayletKiller,
    "KillWorker": WorkerKillerActor,
    "TerminateEC2Instance": EC2InstanceTerminator,
    "TerminateEC2InstanceWithGracePeriod": EC2InstanceTerminatorWithGracePeriod,
}


MAX_HEAD_NODE_MEMORY_BYTES_BY_CASE = {
    "many-tiny-objects": 13 * 1024**3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backpressure benchmark")
    parser.add_argument(
        "--case",
        choices=[
            "fast-producer-slow-consumer",
            "many-tiny-objects",
            "training-prefetch",
            "training-prefetch-single-node",
        ],
        required=True,
    )
    parser.add_argument(
        "--chaos",
        action="store_true",
        default=False,
        help=(
            "Whether to enable chaos. If set, this script periodically kills "
            "worker nodes using the killer selected by --chaos-type."
        ),
    )
    parser.add_argument(
        "--chaos-type",
        choices=list(CHAOS_KILLERS),
        default="TerminateEC2Instance",
        help=(
            "Which chaos killer to use when --chaos is set. Defaults to "
            "TerminateEC2Instance, which terminates the instance with no grace "
            "period so its objects are lost immediately (tests lineage recovery)."
        ),
    )
    parser.add_argument(
        "--chaos-max-kills",
        type=int,
        default=None,
        help=(
            "Max number of resources the chaos killer may kill. Defaults to "
            "unbounded (kills every kill_interval_s for the whole run). Bound this "
            "to keep the cluster from degrading faster than nodes are replaced."
        ),
    )
    parser.add_argument(
        "--recovery-mode",
        choices=["data", "core"],
        default="data",
        help=(
            "Which lineage-recovery backend the chaos run relies on. 'data' "
            "(default) uses Ray Data seed-input lineage recovery and asserts it "
            "fired. 'core' uses Ray Core object reconstruction only -- run with "
            "Data seed-input recovery OFF (do not set "
            "RAY_DATA_ENABLE_SEED_INPUT_LINEAGE_RECOVERY) and Core reconstruction "
            "ON. This is the Core-only control: it skips the Data-recovery "
            "assertions since Core recovers lost objects transparently, and "
            "'lineage_recoveries' will read 0 (any recovery came from Core)."
        ),
    )
    return parser.parse_args()


def make_inputs(num_input_blocks: int):
    return [
        pa.Table.from_pydict({"id": [input_id]}) for input_id in range(num_input_blocks)
    ]


def produce(
    batch,
    *,
    output_batches_per_input_batch: int,
    output_batch_rows: int,
    output_row_bytes: int,
):
    for _ in range(output_batches_per_input_batch):
        yield {
            "data": np.zeros((output_batch_rows, output_row_bytes), dtype=np.uint8),
        }


def consume_slow(batch, *, sleep_s: float):
    time.sleep(sleep_s)
    return {"status": ["ok"]}


def run_fast_producer_slow_consumer():
    """Benchmark backpressure from a fast producer and a slow consumer.

    Produces 1,024 batches of 128 MiB each, for 128 GiB of logical output
    data. A single consumer sleeps for 1 second per batch, creating sustained
    backpressure on the producer.
    """
    num_input_blocks = 128
    output_batches_per_input_batch = 8
    output_batch_rows = 128
    output_row_bytes = 1024**2
    consumer_sleep_s = 1.0

    producer = functools.partial(
        produce,
        output_batches_per_input_batch=output_batches_per_input_batch,
        output_batch_rows=output_batch_rows,
        output_row_bytes=output_row_bytes,
    )
    consumer = functools.partial(consume_slow, sleep_s=consumer_sleep_s)

    ds = (
        ray.data.from_blocks(make_inputs(num_input_blocks))
        .map_batches(producer)
        .map_batches(consumer, compute=ray.data.TaskPoolStrategy(size=1))
    )
    for _ in ds.iter_internal_ref_bundles():
        pass


def run_many_tiny_objects():
    """Benchmark backpressure from many small task outputs.

    Produces 100,000 outputs of about 50 KiB each, or about 4.8 GiB of logical
    output data. The payload size targets Ray's small-object direct-call path,
    while a single consumer sleeps for 10 ms per batch, creating pressure from
    many queued outputs.
    """
    num_input_blocks = 100_000
    output_batches_per_input_batch = 1
    output_batch_rows = 1
    # Stay below Ray's 100 KiB direct-call limit so outputs are sent inline.
    output_row_bytes = 50 * 1024
    consumer_sleep_s = 0.01

    producer = functools.partial(
        produce,
        output_batches_per_input_batch=output_batches_per_input_batch,
        output_batch_rows=output_batch_rows,
        output_row_bytes=output_row_bytes,
    )
    consumer = functools.partial(consume_slow, sleep_s=consumer_sleep_s)

    ds = (
        ray.data.from_blocks(make_inputs(num_input_blocks))
        .map_batches(producer)
        .map_batches(consumer, compute=ray.data.TaskPoolStrategy(size=1))
    )
    for _ in ds.iter_internal_ref_bundles():
        pass


def run_training_prefetch(*, num_trainers: int):
    """Benchmark backpressure from training consumers that prefetch data.

    Produces 1,024 batches of 128 MiB each, for 128 GiB of logical output data,
    then splits them evenly across ``num_trainers`` trainers.

    Each trainer prefetches 8 batches, corresponding to about 1 GiB of data
    per trainer, and sleeps for 1 second after consuming each batch.
    """
    num_input_blocks = 128
    output_batches_per_input_batch = 8
    output_batch_rows = 128
    output_row_bytes = 1024**2
    consumer_sleep_s = 1.0
    prefetch_batches = 8

    producer = functools.partial(
        produce,
        output_batches_per_input_batch=output_batches_per_input_batch,
        output_batch_rows=output_batch_rows,
        output_row_bytes=output_row_bytes,
    )

    trainers = [
        Trainer.options(scheduling_strategy="SPREAD").remote(
            consumer_sleep_s=consumer_sleep_s,
            prefetch_batches=prefetch_batches,
        )
        for _ in range(num_trainers)
    ]

    trainer_node_ids = ray.get([trainer.get_node_id.remote() for trainer in trainers])

    iterators = (
        ray.data.from_blocks(make_inputs(num_input_blocks))
        .map_batches(producer)
        .streaming_split(
            num_trainers,
            equal=True,
            locality_hints=trainer_node_ids,
        )
    )

    ray.get(
        [
            trainers[i].train.remote(iterators[i], batch_size=output_batch_rows)
            for i in range(num_trainers)
        ]
    )


@ray.remote(num_cpus=1)
class Trainer:
    def __init__(self, consumer_sleep_s: float, prefetch_batches: int):
        self._consumer_sleep_s = consumer_sleep_s
        self._prefetch_batches = prefetch_batches

    def train(self, data_iterator, batch_size: int):
        for _ in data_iterator.iter_batches(
            batch_size=batch_size,
            prefetch_batches=self._prefetch_batches,
        ):
            time.sleep(self._consumer_sleep_s)

    def get_node_id(self) -> str:
        return ray.get_runtime_context().get_node_id()


def install_recovery_counter() -> dict:
    """Count Ray Data seed-input lineage recoveries for the current run.

    Wraps ``LineageTracker.register_task_failed``, which the executor calls once
    per detected loss, so the benchmark can assert recovery actually fired under
    chaos. The streaming executor runs on this (driver) process, so patching the
    class here observes its tracker instance.

    Counts only calls that return seed ids. An empty list means reconstruction of
    that partition was already under way and nothing was resubmitted, so counting
    it would overstate how many recoveries actually ran.

    Imported lazily: only the Data seed-input recovery build has this module, so
    importing it at module scope would break the 'core' control on stock Ray.
    """
    import datetime as _dt

    from ray.data._internal.execution import lineage_tracker as lt_mod

    state = {"recoveries": 0}
    orig_register_task_failed = lt_mod.LineageTracker.register_task_failed

    def counting_register_task_failed(self, data_task_id, plan_id=None):
        seed_task_ids, assigned_plan_id = orig_register_task_failed(
            self, data_task_id, plan_id
        )
        if seed_task_ids:
            state["recoveries"] += 1
            print(
                "[RECOVERY_MARK] #%d %s task=%s plan=%s seeds=%d"
                % (
                    state["recoveries"],
                    _dt.datetime.now().isoformat(),
                    data_task_id,
                    assigned_plan_id,
                    len(seed_task_ids),
                ),
                flush=True,
            )
        return seed_task_ids, assigned_plan_id

    lt_mod.LineageTracker.register_task_failed = counting_register_task_failed
    return state


def start_chaos(chaos_type: str = "TerminateEC2Instance", max_to_kill=None):
    assert ray.is_initialized()

    resource_killer_cls = CHAOS_KILLERS[chaos_type]

    head_node_id = ray.get_runtime_context().get_node_id()
    scheduling_strategy = NodeAffinitySchedulingStrategy(
        node_id=head_node_id, soft=False
    )
    # Kill a worker node so its in-store objects are actually lost, rather than
    # drained/migrated. TerminateEC2Instance (the default) terminates the
    # instance with no grace period, so downstream tasks that still need the
    # lost objects must be recovered via Ray Data seed-input lineage recovery.
    # `kill_delay_s` lets the pipeline build a backlog of produced blocks on
    # worker nodes first, so a kill reliably loses objects that downstream tasks
    # still need. Keeps >=1 worker alive automatically (see NodeKillerBase).
    resource_killer = resource_killer_cls.options(
        scheduling_strategy=scheduling_strategy
    ).remote(
        head_node_id,
        kill_interval_s=30,
        kill_delay_s=20,
        max_to_kill=max_to_kill,
    )

    ray.get(resource_killer.ready.remote())

    resource_killer.run.remote()


def main(args: argparse.Namespace):
    benchmark = Benchmark(
        max_head_node_memory_bytes=MAX_HEAD_NODE_MEMORY_BYTES_BY_CASE.get(args.case)
    )

    if args.case == "fast-producer-slow-consumer":
        case_fn = functools.partial(run_fast_producer_slow_consumer)
    elif args.case == "many-tiny-objects":
        case_fn = functools.partial(run_many_tiny_objects)
    elif args.case == "training-prefetch":
        case_fn = functools.partial(run_training_prefetch, num_trainers=8)
    elif args.case == "training-prefetch-single-node":
        case_fn = functools.partial(run_training_prefetch, num_trainers=1)
    else:
        raise ValueError(f"Unexpected benchmark case: {args.case}")

    recovery_state = None
    if args.chaos and args.recovery_mode == "data":
        assert ray.data.DataContext.get_current().enable_seed_input_lineage_recovery, (
            "Chaos is enabled but seed-input lineage recovery is off. Run with "
            "RAY_DATA_ENABLE_SEED_INPUT_LINEAGE_RECOVERY=1, or pass "
            "--recovery-mode core to measure Ray Core reconstruction instead."
        )
        # Only Data seed-input recovery has a LineageTracker to count. In 'core'
        # mode there is nothing to patch: Core recovers lost objects transparently.
        recovery_state = install_recovery_counter()

    def run_case():
        # Started inside `run_fn`, where Ray is up: `start_chaos` needs the runtime
        # context to pin the killer actor to the head node.
        if args.chaos:
            start_chaos(args.chaos_type, max_to_kill=args.chaos_max_kills)
        case_fn()
        if recovery_state is None:
            return {}
        # The case completing proves the dataset finished despite chaos; this
        # asserts it finished *because of* seed-input lineage recovery (an object
        # owner was killed and the executor resubmitted its seed input), rather
        # than chaos never losing a needed object.
        assert recovery_state["recoveries"] > 0, (
            "Chaos was enabled but Ray Data seed-input lineage recovery never "
            "fired -- the test did not exercise lineage recovery."
        )
        return {"lineage_recoveries": recovery_state["recoveries"]}

    benchmark.run_fn(args.case, run_case)
    benchmark.write_result()


if __name__ == "__main__":
    main(parse_args())
