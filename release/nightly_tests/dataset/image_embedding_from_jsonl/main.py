# ABOUTME: Ray Data image embedding benchmark (JSONL input) with GPU and CPU profiling.
# ABOUTME: Reads base64-encoded images from JSONL, runs HuggingFace ViT inference on GPU actors, writes to parquet.

from __future__ import annotations

import argparse
import os
import time
import uuid
from io import BytesIO
from typing import Any, Dict, List

import numpy as np
import ray
import ray.data
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from pybase64 import b64decode

from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray._private.test_utils import RayletKiller
from ray.job_config import JobConfig
from benchmark import (
    Benchmark,
    RuntimeEnvSetupTracker,
    benchmark_py_modules,
    collect_dataset_stats,
)
from profiling.coordinator import Profiling
from profiling import nvtx as profiling_nvtx
from profiling.metrics import extract_pipeline_metrics


INPUT_PREFIX = "s3://ray-benchmark-data-internal-us-west-2/10TiB-jsonl-images"
OUTPUT_PREFIX = f"s3://ray-data-write-benchmark/{uuid.uuid4().hex}"

BATCH_SIZE = 1024
# Ray Data can't prevent OOMs if you don't set `memory` for high-memory operations like
# this one. We chose 3 GiB because it was the max USS we observed in Ray 2.56 weekly
# test runs.
READ_MEMORY = 3 * 1024**3

# Fake-GPU emulation: sleep instead of running the real ViT forward pass so the
# benchmark can run on CPU nodes carrying a custom GPU:1 resource (sidesteps real
# GPU capacity limits). 2.5 s/batch of 1024 is the measured ViT-base inference
# time on an A10G (g5.4xlarge) from original GPU based test
GPU_SECONDS_PER_IMAGE = 2.5 / 1024

PROCESSOR = ViTImageProcessor(
    do_convert_rgb=None,
    do_normalize=True,
    do_rescale=True,
    do_resize=True,
    image_mean=[0.5, 0.5, 0.5],
    image_std=[0.5, 0.5, 0.5],
    resample=2,
    rescale_factor=0.00392156862745098,
    size={"height": 224, "width": 224},
)

# Node-kill intensity for --chaos-scale on the 140-node cluster. Each value is
# (kill_interval_s, kill_delay_s, max_to_kill).
CHAOS_SCALES = {
    "low": (60, 60, 2),
    "medium": (60, 60, 5),
    "high": (30, 60, 15),
}

JOB_ID = os.environ.get("ANYSCALE_JOB_ID", f"local-{uuid.uuid4().hex[:8]}")
SHARED_OUTDIR = f"/mnt/shared_storage/image_embedding_jsonl/{JOB_ID}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inference-concurrency",
        nargs=2,
        type=int,
        required=True,
        help="The minimum and maximum concurrency for the inference operator.",
    )
    parser.add_argument(
        "--chaos",
        action="store_true",
        help=(
            "Whether to enable chaos. If set, this script kills worker raylets at "
            "the rate set by --chaos-scale."
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
        "--fake-gpu",
        action="store_true",
        help=(
            "Use fake gpu mode if set. In this mode, Infer uses CPU nodes and sleeps "
            "for the time GPU based Infer takes to process one batch"
        ),
    )
    parser.add_argument(
        "--verify-output",
        action="store_true",
        help=(
            "Read with include_paths and check per-file row conservation against the "
            "source corpus after the write (see verify_output.py), failing the run on "
            "a mismatch. Off by default so the stock benchmark path is unchanged."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline UDFs
# ---------------------------------------------------------------------------


def decode(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    image_data = b64decode(row["image"], None, True)
    image = Image.open(BytesIO(image_data))
    width, height = image.size
    decoded = {
        "original_url": row["url"],
        "original_width": width,
        "original_height": height,
        "image": np.asarray(image),
    }
    # Presence-driven: the read only supplies `path` under --verify-output, so no
    # flag has to be propagated to the workers. This is the only per-row identity
    # available -- the corpus is one image and one url replicated, so `original_url`
    # is not a key and no per-row id can be injected.
    if "path" in row:
        decoded["source_file"] = row["path"]
    return [decoded]


def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
    outputs = PROCESSOR(images=row["image"])["pixel_values"]
    assert len(outputs) == 1, len(outputs)
    row["image"] = outputs[0]
    return row


class Infer:
    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224"
        ).to(self._device)

        self._call_count = 0
        self._profiling_active = False
        self._profiler_done = False
        self._profiler_mode = os.environ.get("PROFILER_MODE", "none")
        self._skip_batches = int(os.environ.get("PROFILE_SKIP_BATCHES", "0"))
        self._active_batches = int(os.environ.get("PROFILE_ACTIVE_BATCHES", "10000"))
        self._node_ip = ray.util.get_node_ip_address()

        # The capture range opens at the first cuda_profiler_fence call
        # (batch == skip_batches + 1) and normally never closes via the
        # in-loop fence (active_batches is set high). Atexit closes it on
        # interpreter shutdown; with capture-range-end:stop in
        # nsys_runtime_env() that finalizes the .nsys-rep synchronously
        # before Ray tears the actor down.
        if self._profiler_mode == "nsys":
            import atexit

            def _stop_nsys():
                if self._profiling_active:
                    torch.cuda.cudart().cudaProfilerStop()
                    self._profiling_active = False

            atexit.register(_stop_nsys)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self._call_count += 1

        # --- nsys capture range control via CUDA profiler API ---
        if self._profiler_mode == "nsys" and not self._profiler_done:
            result = profiling_nvtx.cuda_profiler_fence(
                self._call_count,
                self._skip_batches,
                self._active_batches,
                self._node_ip,
            )
            if result[0] is not None:
                self._profiling_active = result[0]
            if result[1] is not None:
                self._profiler_done = result[1]

        # --- GPU work (with NVTX annotations when nsys is active) ---
        if self._profiler_mode == "nsys":
            with profiling_nvtx.profiling_range(f"MapBatches_call_{self._call_count}"):
                with profiling_nvtx.profiling_range("h2d_transfer"):
                    next_tensor = torch.from_numpy(batch["image"]).to(
                        dtype=torch.float32,
                        device=self._device,
                        non_blocking=True,
                    )

                with profiling_nvtx.profiling_range("inference"):
                    with torch.inference_mode():
                        output = self._model(next_tensor).logits

                with profiling_nvtx.profiling_range("d2h_postprocess"):
                    result = {
                        "original_url": batch["original_url"],
                        "original_width": batch["original_width"],
                        "original_height": batch["original_height"],
                        "output": output.cpu().numpy(),
                    }
                    if "source_file" in batch:
                        result["source_file"] = batch["source_file"]
        else:
            next_tensor = torch.from_numpy(batch["image"]).to(
                dtype=torch.float32, device=self._device, non_blocking=True
            )
            with torch.inference_mode():
                output = self._model(next_tensor).logits
            result = {
                "original_url": batch["original_url"],
                "original_width": batch["original_width"],
                "original_height": batch["original_height"],
                "output": output.cpu().numpy(),
            }
            if "source_file" in batch:
                result["source_file"] = batch["source_file"]

        return result


class FakeInfer:
    """Fake-GPU Infer: sleeps for the measured per-batch inference time instead
    of running the real ViT forward pass, so the benchmark can run on CPU nodes
    carrying a custom GPU:1 resource (see GPU_SECONDS_PER_IMAGE)."""

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        n = len(batch["original_url"])
        # Emulate the GPU forward pass: sleep for the batch's inference time.
        time.sleep(GPU_SECONDS_PER_IMAGE * n)
        result = {
            "original_url": batch["original_url"],
            "original_width": batch["original_width"],
            "original_height": batch["original_height"],
            # Real Infer returns model(...).logits, shape (n, 1000) float32.
            "output": np.zeros((n, 1000), dtype=np.float32),
        }
        if "source_file" in batch:
            result["source_file"] = batch["source_file"]
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace, profiling: Profiling):
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

    # `default_map_logical_memory_enabled` is a best practice that's required for
    # Ray Data to prevent OOMs. It's not enabled by default in Ray 2.56, but we
    # intend to enable it by default in a future release.
    ray.data.DataContext.get_current().default_map_logical_memory_enabled = True

    infer_cls = FakeInfer if args.fake_gpu else Infer
    infer_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_gpus": 1,
        "concurrency": tuple(args.inference_concurrency),
    }
    if args.fake_gpu:
        # Fake-GPU nodes are CPU nodes whose GPU:1 is a custom resource, so the
        # actor must also claim a real CPU for its sleep "compute".
        infer_kwargs["num_cpus"] = 1

    nsys_env = profiling.nsys_runtime_env()
    if nsys_env:
        infer_kwargs["runtime_env"] = nsys_env

    num_gpus = max(args.inference_concurrency)
    ds_holder = {}
    # `source_file` is the only identity this corpus admits, and it costs a column
    # on every row, so it is only requested when the output is going to be checked.
    read_kwargs = {"include_paths": True} if args.verify_output else {}

    def benchmark_fn():
        ds = (
            ray.data.read_json(
                INPUT_PREFIX, lines=True, memory=READ_MEMORY, **read_kwargs
            )
            .flat_map(decode)
            .map(preprocess)
            .map_batches(
                infer_cls,
                **infer_kwargs,
            )
        )
        ds.write_parquet(OUTPUT_PREFIX)
        ds_holder["ds"] = ds

    benchmark.run_fn("main", benchmark_fn)

    ds = ds_holder["ds"]
    # Hold ds in scope so Ray Data keeps the actor pool alive while nsys
    # finalizes its .nsys-rep files via stop-on-exit / atexit. Without
    # this, ds drops out of scope on return and Ray tears the actors
    # down before nsys gets to flush.
    if profiling.profiler_mode == "nsys":
        print("Holding ds in scope for 30s to let nsys finalize...", flush=True)
        time.sleep(30)

    metrics = collect_dataset_stats(ds)
    metrics["runtime_env_setup"] = RuntimeEnvSetupTracker.collect()
    if profiling.is_enabled():
        metrics.update(
            extract_pipeline_metrics(ds, num_gpus=num_gpus, outdir=SHARED_OUTDIR)
        )
    benchmark.result["main"].update(metrics)

    # Recorded before `write_result` so the count actually reaches result.json.
    if recovery_state is not None:
        benchmark.result["main"]["lineage_recoveries"] = recovery_state["recoveries"]

    benchmark.write_result()

    if recovery_state is not None:
        # Asserted after `write_result` so the count is still reported on failure.
        assert recovery_state["recoveries"] > 0, (
            "Chaos was enabled but Ray Data lineage reconstruction never fired, so "
            "this run did not exercise recovery."
        )

    if args.verify_output:
        # Verified with pyarrow rather than Ray Data: checking Ray Data's output
        # with Ray Data would let a bug cancel itself out.
        from verify_output import verify

        print(f"Verifying sink {OUTPUT_PREFIX}")
        if verify(OUTPUT_PREFIX) != 0:
            raise RuntimeError(
                f"Output verification failed for {OUTPUT_PREFIX}; see the checks above."
            )


def install_recovery_counter() -> dict:
    """Count Ray Data lineage reconstructions for this run.

    Wraps ``LineageTracker.register_task_failed``, which the executor calls once
    per detected loss. The streaming executor runs on this (driver) process, so
    patching the class here observes its tracker instance. Counts only calls that
    return seed ids.
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
    # `RayletKiller`, not a terminator with a grace period: a graceful terminator
    # drains the node, so nothing is lost and reconstruction never runs.
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

    # S3 sometimes returns transient ACCESS_DENIED on HeadObject under heavy
    # concurrent load (credential refresh or throttling). Retry these instead
    # of aborting the entire job.
    ctx = ray.data.DataContext.get_current()
    ctx.retried_io_errors = list(ctx.retried_io_errors) + [
        "AWS Error ACCESS_DENIED",
    ]

    num_gpu_nodes = max(args.inference_concurrency)
    profiling = Profiling(outdir=SHARED_OUTDIR, num_gpu_nodes=num_gpu_nodes)

    profiling.start(
        extra_config={
            "RAY_COMMIT": ray.__commit__,
            "INFERENCE_CONCURRENCY": args.inference_concurrency,
        }
    )
    try:
        main(args, profiling)
    finally:
        profiling.stop(s3_prefix=f"image-embedding-jsonl/{JOB_ID}")
