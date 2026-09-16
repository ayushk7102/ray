"""Output correctness check for ``image_embedding_from_jsonl``.

The ceiling here is per-file, not per-row. The corpus is a single image and a
single url replicated millions of times, so ``original_url`` looks like a key but
is not one: rows sampled from different files are byte-identical. No per-row
identity exists in the source and none can be injected, because an injected
counter would be re-stamped by a re-executed task and prove nothing.

What *is* available is the file each row came from (``--verify-output`` reads with
``include_paths=True`` and ``decode`` carries it through as ``source_file``). That
gives per-file row conservation, which still catches a loss in one file masked by
a duplicate in another -- the failure a bare total-row count would hide.

Ground truth is analytic, so verifying does not require scanning the 10 TiB input:
every record in this corpus is a fixed width, so ``rows = file_size / record_width``.
The width is re-derived from live S3 object sizes rather than hardcoded, and the
check refuses to run if any input file is not a whole number of records -- that
premise failing means the corpus changed and the expectations are stale.

Verified with pyarrow, never Ray Data: checking Ray Data's output with Ray Data
would let a bug cancel itself out.
"""

import argparse
import sys
from collections import Counter
from typing import Dict, Tuple

import pyarrow.dataset as ds
import pyarrow.fs as pafs

INPUT_PREFIX = "s3://ray-benchmark-data-internal-us-west-2/10TiB-jsonl-images"

# Only the identity column. The embedding is 1000 float32s per row; reading it
# would turn a small scan into a very large one.
KEY_COLUMN = "source_file"


def _strip_scheme(uri: str) -> str:
    return uri.split("://", 1)[1] if "://" in uri else uri


def input_file_sizes(prefix: str) -> Dict[str, int]:
    """Size in bytes of every input file, keyed by path without scheme."""
    fs, root = pafs.FileSystem.from_uri(prefix)
    selector = pafs.FileSelector(root, recursive=True)
    sizes = {}
    for info in fs.get_file_info(selector):
        if info.type == pafs.FileType.File and info.size > 0:
            sizes[info.path] = info.size
    return sizes


def derive_record_width(prefix: str, sample_path: str) -> int:
    """The fixed per-record byte width, measured from one record in the corpus.

    Every JSONL record here is the same width -- one replicated image plus a
    fixed-length url -- so reading the first line of any file gives the width for
    all of them. It is measured rather than hardcoded so that a corpus change is
    caught by the whole-number check below instead of silently shifting every
    expectation.

    Deliberately not the GCD of the file sizes: every file in this corpus is the
    same size, so the GCD is the file size itself, which would imply one record
    per file and make every count vacuously wrong.
    """
    fs, _ = pafs.FileSystem.from_uri(prefix)
    # One record is ~1.5 MB; read a bounded window and take the first newline.
    with fs.open_input_stream(sample_path) as handle:
        window = handle.read(8 * 1024 * 1024)
    newline = window.find(b"\n")
    if newline < 0:
        return 0
    return newline + 1


def expected_rows_per_file(sizes: Dict[str, int], width: int) -> Dict[str, int]:
    return {path: size // width for path, size in sizes.items()}


def actual_rows_per_file(sink: str) -> Counter:
    dataset = ds.dataset(
        _strip_scheme(sink), format="parquet", filesystem=pafs.S3FileSystem()
    )
    counts = Counter()
    for batch in dataset.to_batches(columns=[KEY_COLUMN]):
        for value in batch.column(KEY_COLUMN).to_pylist():
            counts[_strip_scheme(value) if value is not None else None] += 1
    return counts


def _fmt(items, limit: int = 10) -> str:
    items = list(items)
    head = ", ".join(str(i) for i in items[:limit])
    return head + (f", ... (+{len(items) - limit} more)" if len(items) > limit else "")


def verify(sink: str, prefix: str = INPUT_PREFIX) -> int:
    checks: list[Tuple[str, bool, str]] = []

    sizes = input_file_sizes(prefix)
    checks.append(
        ("input files found", bool(sizes), f"{len(sizes)} files under {prefix}")
    )
    if not sizes:
        _report(checks)
        return 1

    width = derive_record_width(prefix, min(sizes))
    ragged = [p for p, s in sizes.items() if width == 0 or s % width]
    # The premise the whole analytic ground truth rests on. Fail loudly rather
    # than compute expectations from a corpus that no longer matches.
    checks.append(
        (
            "every input file is a whole number of records",
            not ragged and width > 0,
            f"record width {width} B; ragged files: {_fmt(ragged)}"
            if ragged
            else f"record width {width} B",
        )
    )
    if ragged or width == 0:
        _report(checks)
        return 1

    expected = expected_rows_per_file(sizes, width)
    actual = actual_rows_per_file(sink)

    unknown = sorted(set(actual) - set(expected))
    checks.append(("no rows from unknown files", not unknown, _fmt(unknown)))

    missing_files = sorted(set(expected) - set(actual))
    checks.append(
        ("every input file reached the sink", not missing_files, _fmt(missing_files))
    )

    short = {
        p: (expected[p], actual[p])
        for p in expected
        if p in actual and actual[p] < expected[p]
    }
    checks.append(
        (
            "no file lost rows",
            not short,
            _fmt(f"{p}: {e} expected, {a} present" for p, (e, a) in short.items()),
        )
    )

    over = {
        p: (expected[p], actual[p])
        for p in expected
        if p in actual and actual[p] > expected[p]
    }
    # A duplicate is the specific failure OBJECT_PRUNED exists to prevent: an
    # output that was never lost being re-emitted by a reconstruction.
    checks.append(
        (
            "no file gained rows",
            not over,
            _fmt(f"{p}: {e} expected, {a} present" for p, (e, a) in over.items()),
        )
    )

    exp_total, act_total = sum(expected.values()), sum(actual.values())
    checks.append(
        (
            "total row count matches",
            exp_total == act_total,
            f"{act_total} present, {exp_total} expected",
        )
    )

    return _report(checks)


def _report(checks) -> int:
    failed = 0
    for name, ok, detail in checks:
        print(
            f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else "")
        )
        failed += not ok
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", help="The sink `main.py` wrote to, e.g. s3://bucket/hex"
    )
    parser.add_argument(
        "--input-prefix",
        default=INPUT_PREFIX,
        help="Source corpus to derive expectations from.",
    )
    args = parser.parse_args()
    return verify(args.path, args.input_prefix)


if __name__ == "__main__":
    sys.exit(main())
