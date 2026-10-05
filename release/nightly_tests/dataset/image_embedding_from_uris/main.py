import argparse
import io
import os
import uuid
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from benchmark import (
    Benchmark,
    RuntimeEnvSetupTracker,
    benchmark_py_modules,
    collect_dataset_stats,
)
from PIL import Image
from torchvision.models import vit_b_16, ViT_B_16_Weights
import albumentations as A
import ray
import copy
import itertools
from typing import List
import string
import random
import time
from ray.data.expressions import download
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray._private.test_utils import RayletKiller
from ray.job_config import JobConfig

# Node-kill intensity for --chaos-scale on the 100-node GPU cluster. Medium is the
# original setting. Each value is (kill_interval_s, kill_delay_s, max_to_kill).
CHAOS_SCALES = {
    "low": (60, 60, 2),
    "medium": (60, 60, 5),
    "high": (30, 60, 15),
}


WRITE_PATH = os.environ.get("RAY_DATA_WRITE_PATH") or (
    f"s3://ray-data-write-benchmark/{uuid.uuid4().hex}"
)
BUCKET = "ray-benchmark-data-internal-us-west-2"

# Assumptions: homogenously shaped images, homogenous images
# Each image is 2048 * 2048 * 3 = 12.58 MB -> 11 images / block. 8 blocks per task, so ~88 images per task.
IMAGES_PER_BLOCK = 11
BLOCKS_PER_TASK = 8
NUM_UNITS = 1380
NUM_CONTAINERS = 50
OVERRIDE_NUM_BLOCKS = int(NUM_CONTAINERS * NUM_UNITS / IMAGES_PER_BLOCK)
PATCH_SIZE = 256

# Largest batch that can fit on a T4.
BATCH_SIZE = 1200

# On a T4 GPU, it takes ~11.3s to perform inference on 1200 images. So, the time per
# image is 11.3s / 1200 ~= 0.0094s.
INFERENCE_LATENCY_PER_IMAGE_S = 0.0094


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference-concurrency",
        nargs=2,
        type=int,
        required=True,
        help="The minimum and maximum concurrency for the inference operator.",
    )
    parser.add_argument(
        "--sf",
        dest="scale_factor",
        type=int,
        default=1,
        help=(
            "The number of copies of the dataset to read. Use this to simulate a larger "
            "dataset."
        ),
    )
    parser.add_argument(
        "--chaos",
        action="store_true",
        help=(
            "Whether to enable chaos. If set, this script terminates one worker node "
            "every minute with a grace period."
        ),
    )
    parser.add_argument(
        "--chaos-scale",
        choices=sorted(CHAOS_SCALES),
        default="medium",
        help="Node-kill intensity when --chaos is set. See CHAOS_SCALES.",
    )
    parser.add_argument(
        "--recovery-mode",
        choices=["data", "core"],
        default="core",
        help=(
            "'data' starts the driver with "
            "JobConfig(_disable_job_level_lineage_reconstruction=True), so Ray Data "
            "reconstructs lost objects and Core does not, and asserts recovery "
            "fired under chaos. 'core' keeps the default job config."
        ),
    )
    parser.add_argument(
        "--verify-output",
        action="store_true",
        help=(
            "After the write, check the sink against the analytically-known key set "
            "(see verify_output.py) and fail the run if it does not match. Off by "
            "default so the stock benchmark path is unchanged."
        ),
    )
    return parser.parse_args()


def create_metadata(scale_factor: int):
    # TODO(mowen): Handle repeats of the dataset if scale_factor > 1
    # simulate various text metadata fields alongside image metadata
    metadata = pd.DataFrame(
        [
            {
                "metadata_0": "".join(random.choices(string.ascii_letters, k=16)),
                "metadata_1": "".join(random.choices(string.ascii_letters, k=16)),
                "metadata_2": "".join(random.choices(string.ascii_letters, k=16)),
                "metadata_3": "".join(random.choices(string.ascii_letters, k=16)),
                "metadata_4": "".join(random.choices(string.ascii_letters, k=16)),
                "metadata_5": "".join(random.choices(string.ascii_letters, k=16)),
                "metadata_6": "".join(random.choices(string.ascii_letters, k=16)),
                "container_order_read_id": f"{i:04d}_{j:04d}",
                "container_id": i,
                "channel0_uris": f"s3://{BUCKET}/15TiB-high-resolution-images/group={i:04d}/{j:04d}_{0}.png",
                "channel1_uris": f"s3://{BUCKET}/15TiB-high-resolution-images/group={i:04d}/{j:04d}_{1}.png",
                "channel2_uris": f"s3://{BUCKET}/15TiB-high-resolution-images/group={i:04d}/{j:04d}_{2}.png",
                "applied_scale": 1,
            }
            for j in range(NUM_UNITS)
            for i in range(NUM_CONTAINERS)
        ]
    )
    # Dense positional id over the frame above, so `(row_serial, patch_x, patch_y)`
    # is a primary key over the sink. Read from the source ordering rather than
    # stamped per attempt, so a re-executed task reproduces the same values --
    # a per-attempt counter would make output verification vacuous.
    metadata["row_serial"] = np.arange(len(metadata), dtype=np.int64)
    return metadata


def combine_channels(row: Dict[str, Any]) -> Dict[str, np.ndarray]:
    channels = []
    for i in range(3):
        data = io.BytesIO(row.pop(f"channel{i}"))
        image = Image.open(data)
        channels.append(np.array(image))

    row["image"] = np.dstack(channels)

    return row


def process_image(row: Dict[str, Any]) -> Dict[str, np.ndarray]:
    transform = A.Compose(
        [
            A.ToFloat(),
            A.LongestMaxSize(
                max_size=int(row["image"].shape[0] * float(1.0 / row["applied_scale"]))
            ),
            A.FromFloat(dtype="uint8"),
        ]
    )
    row["image"] = transform(image=row["image"])["image"]
    return row


def patch_image(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    image = row.pop("image")

    patches = []
    width, height, _ = image.shape
    for x, y in itertools.product(
        range(PATCH_SIZE, width - PATCH_SIZE, PATCH_SIZE),
        range(PATCH_SIZE, height - PATCH_SIZE, PATCH_SIZE),
    ):
        patch = image[y : y + PATCH_SIZE, x : x + PATCH_SIZE, :]

        patch_row = copy.deepcopy(row)
        patch_row["patch_x"] = x
        patch_row["patch_y"] = y
        patch_row["patch_width"] = PATCH_SIZE
        patch_row["patch_height"] = PATCH_SIZE
        patch_row["patch"] = patch

        patches.append(patch_row)

    return patches


class ProcessPatches:
    def __init__(self, transform):
        self._transform = transform

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        batch["patch"] = self._transform(
            torch.as_tensor(batch["patch"]).permute(0, 3, 1, 2)
        )
        return batch


class EmbedPatches:
    def __init__(self, model, device):
        self._model = ray.get(model)
        self._model.eval()
        self._model.to(device)
        self._device = device

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        inputs = torch.as_tensor(batch.pop("patch"), device=self._device)
        with torch.inference_mode():
            output = self._model(inputs)
            batch["embedding"] = output.cpu().numpy()
            return batch


class FakeEmbedPatches:
    def __init__(self, model, device):
        self._model = ray.get(model)
        self._model.eval()

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        inputs = torch.as_tensor(batch.pop("patch"))
        with torch.inference_mode():
            # Simulate inference latency with a sleep
            time.sleep(INFERENCE_LATENCY_PER_IMAGE_S * len(inputs))
            # Generate fake embeddings
            output = torch.rand((len(inputs), 1000), dtype=torch.float)
            batch["embedding"] = output.cpu().numpy()
            return batch


def main(args: argparse.Namespace):
    benchmark = Benchmark()

    if args.chaos:
        start_chaos(args.chaos_scale)

    recovery_state = None
    if args.chaos and args.recovery_mode == "data":
        assert ray.data.DataContext.get_current().enable_ray_data_reconstruction, (
            "Chaos is enabled with --recovery-mode data, but Ray Data "
            "reconstruction is off. Check that the driver started with "
            "JobConfig(_disable_job_level_lineage_reconstruction=True)."
        )
        recovery_state = install_recovery_counter()

    print("Creating metadata")
    metadata = create_metadata(scale_factor=args.scale_factor)

    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    transform = weights.transforms()
    model_ref = ray.put(model)

    ds_holder = {}

    def benchmark_fn():
        ds = (
            ray.data.from_pandas(metadata)
            .with_column("channel0", download("channel0_uris"))
            .with_column("channel1", download("channel1_uris"))
            .with_column("channel2", download("channel2_uris"))
            .map(combine_channels)
            .filter(lambda row: row["image"].size != 0)
            .map(process_image)
            .flat_map(patch_image)
            .map_batches(ProcessPatches(transform), batch_size="auto")
            .map_batches(
                EmbedPatches,
                num_gpus=1,
                batch_size=BATCH_SIZE,
                concurrency=tuple(args.inference_concurrency),
                fn_constructor_kwargs={"model": model_ref, "device": "cuda"},
            )
        )
        ds.write_parquet(WRITE_PATH)
        ds_holder["ds"] = ds

    benchmark.run_fn("main", benchmark_fn)

    metrics = collect_dataset_stats(ds_holder["ds"])
    metrics["runtime_env_setup"] = RuntimeEnvSetupTracker.collect()
    benchmark.result["main"].update(metrics)

    # Recorded before `write_result` so the count actually reaches result.json.
    if recovery_state is not None:
        benchmark.result["main"]["lineage_recoveries"] = recovery_state["recoveries"]

    benchmark.write_result()

    if recovery_state is not None:
        # The run completing proves the dataset survived chaos; this asserts it
        # survived *because of* recovery. Without it a run where chaos never lost a
        # needed object passes silently and verifies nothing about reconstruction.
        # Asserted after `write_result` so the count is still reported on failure.
        assert recovery_state["recoveries"] > 0, (
            "Chaos was enabled but Ray Data seed-input lineage recovery never fired "
            "-- no object the pipeline needed was lost, so this run did not exercise "
            "recovery."
        )

    if args.verify_output:
        # Verified with pyarrow rather than Ray Data: checking Ray Data's output with
        # Ray Data would let a bug cancel itself out.
        from verify_output import verify

        print(f"Verifying sink {WRITE_PATH}")
        if verify(WRITE_PATH, expect_rows_missing_ok=False) != 0:
            raise RuntimeError(
                f"Output verification failed for {WRITE_PATH}; see the checks above."
            )


def install_recovery_counter() -> dict:
    """Count Ray Data seed-input lineage recoveries for this run.

    Wraps ``LineageTracker.register_task_failed``, which the executor calls once
    per detected loss, so the run can assert recovery actually fired rather than
    passing because chaos never lost anything. The streaming executor runs on this
    (driver) process, so patching the class here observes its tracker instance.

    Counts only calls that return seed ids: an empty list means reconstruction of
    that lineage was already under way and nothing was resubmitted.
    """
    from ray.data._internal.execution import lineage_tracker as lt_mod

    state = {"recoveries": 0}
    original = lt_mod.LineageTracker.register_task_failed

    def counting_register_task_failed(self, data_task_id, plan_id=None):
        seed_task_ids, assigned_plan_id = original(self, data_task_id, plan_id)
        if seed_task_ids:
            state["recoveries"] += 1
        return seed_task_ids, assigned_plan_id

    lt_mod.LineageTracker.register_task_failed = counting_register_task_failed
    return state


def start_chaos(chaos_scale: str = "medium"):
    assert ray.is_initialized()

    head_node_id = ray.get_runtime_context().get_node_id()
    scheduling_strategy = NodeAffinitySchedulingStrategy(
        node_id=head_node_id, soft=False
    )
    # `RayletKiller`, not an EC2 terminator: killing the raylet takes the node's
    # object store down with it, so the objects it held are lost immediately.
    # A terminator *with a grace period* drains the node instead, which is why the
    # previous run finished with zero `ObjectLostError` and never exercised
    # recovery at all -- the pipeline simply never lost anything.
    #
    # `kill_delay_s` lets the pipeline build a backlog of produced blocks first, so
    # a kill reliably takes out objects that downstream tasks still need rather
    # than landing before there is anything to lose. `max_to_kill` bounds the
    # damage so the cluster is not degraded faster than nodes are replaced.
    # `NodeKillerBase` keeps at least one worker alive on its own.
    kill_interval_s, kill_delay_s, max_to_kill = CHAOS_SCALES[chaos_scale]
    print(
        f"[CHAOS] RayletKiller scale={chaos_scale} interval={kill_interval_s}s "
        f"delay={kill_delay_s}s max_to_kill={max_to_kill}",
        flush=True,
    )
    resource_killer = RayletKiller.options(
        scheduling_strategy=scheduling_strategy
    ).remote(
        head_node_id,
        kill_interval_s=kill_interval_s,
        kill_delay_s=kill_delay_s,
        max_to_kill=max_to_kill,
    )

    ray.get(resource_killer.ready.remote())

    resource_killer.run.remote()


if __name__ == "__main__":
    args = parse_args()
    init_kwargs = {}
    if args.recovery_mode == "data":
        # Ray Data reconstruction is gated on this job config. It also turns Core
        # lineage reconstruction off for the job.
        init_kwargs["job_config"] = JobConfig(
            _disable_job_level_lineage_reconstruction=True
        )
    ray.init(runtime_env={"py_modules": benchmark_py_modules()}, **init_kwargs)
    main(args)
