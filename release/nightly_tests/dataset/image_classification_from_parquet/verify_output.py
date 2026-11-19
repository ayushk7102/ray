"""Output correctness check for ``image_classification_from_parquet``.

    python verify_output.py s3://ray-data-write-benchmark/<hex> \
        --data-directory 300G-image-data-synthetic-raw-parquet

Every image in this dataset is byte-identical, so the image is no row id. The
``label`` column is a per-row filename (``dog.jpg``, ``dog_1.jpg``, ...). It repeats
across files but is unique within one, so ``(file, label)`` is a primary key. Both
halves are read from the source, so a re-executed task reproduces them exactly.

``main.py --verify-output`` reads with ``include_paths=True`` and carries ``path``
and ``label`` to the sink. This script compares the multiset of ``(file, label)``
pairs in the sink against the source:

- every source row arrived (dropped rows)
- no row arrived twice (rows from over-reconstruction)
- no unknown row arrived (a reconstructed block carrying the wrong rows)
- per-file row counts match, so a loss in one file can't hide behind a duplicate
  in another

Verified with pyarrow, not Ray Data, so a Ray Data bug can't hide itself. The
source scan reads only the ``label`` column, so the image bytes are never fetched.
"""

import argparse
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

SOURCE_BUCKET = "air-example-data-2"
SOURCE_REGION = "us-west-2"

Pair = Tuple[str, str]


def _source_filesystem() -> pafs.S3FileSystem:
    return pafs.S3FileSystem(anonymous=True, region=SOURCE_REGION)


def _labels_in(path: str) -> List[Pair]:
    # One filesystem per call. Sharing one across threads has crashed at teardown.
    table = pq.read_table(path, columns=["label"], filesystem=_source_filesystem())
    name = os.path.basename(path)
    return [(name, label) for label in table.column("label").to_pylist()]


def source_pairs(data_directory: str) -> Counter:
    root = f"{SOURCE_BUCKET}/{data_directory}"
    listing = _source_filesystem().get_file_info(
        pafs.FileSelector(root, recursive=True)
    )
    files = [f.path for f in listing if f.is_file and f.path.endswith(".parquet")]
    print(f"Reading the label column of {len(files)} source files", flush=True)
    pairs = Counter()
    with ThreadPoolExecutor(32) as pool:
        for file_pairs in pool.map(_labels_in, files):
            pairs.update(file_pairs)
    return pairs


def sink_pairs(sink: str) -> Counter:
    pairs = Counter()
    dataset = ds.dataset(sink, format="parquet")
    for batch in dataset.to_batches(columns=["path", "label"]):
        paths = batch.column("path").to_pylist()
        labels = batch.column("label").to_pylist()
        pairs.update(
            (os.path.basename(path.rstrip("/")), label)
            for path, label in zip(paths, labels)
        )
    return pairs


def _fmt(items, limit: int = 10) -> str:
    items = list(items)
    head = ", ".join(str(i) for i in items[:limit])
    return head + (f", ... (+{len(items) - limit} more)" if len(items) > limit else "")


def verify(sink: str, data_directory: str) -> int:
    expected = source_pairs(data_directory)
    arrived = sink_pairs(sink)
    expected_total, arrived_total = sum(expected.values()), sum(arrived.values())
    print(
        f"source: {expected_total:,} rows, {len(expected):,} distinct (file, label)\n"
        f"sink:   {arrived_total:,} rows, {len(arrived):,} distinct (file, label)\n"
    )

    failures = []

    def check(ok: bool, name: str, detail: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
        if not ok:
            print(f"         {detail}", flush=True)
            failures.append(name)

    check(
        len(expected) == expected_total,
        "(file, label) is unique in the source",
        f"{expected_total - len(expected):,} collisions, so it is no row key",
    )
    check(
        arrived_total == expected_total,
        "total row count",
        f"got {arrived_total:,}, expected {expected_total:,}",
    )
    missing = sorted(set(expected) - set(arrived))
    check(
        not missing,
        "every source row arrived",
        f"{len(missing):,} missing: {_fmt(missing)}",
    )
    unknown = sorted(set(arrived) - set(expected))
    check(not unknown, "no unknown rows", f"{len(unknown):,} unknown: {_fmt(unknown)}")
    duplicated = sorted(pair for pair, n in arrived.items() if n > 1)
    check(
        not duplicated,
        "no row arrived twice",
        f"{len(duplicated):,} duplicated: {_fmt(duplicated)}",
    )
    expected_per_file, arrived_per_file = Counter(), Counter()
    for (name, _), n in expected.items():
        expected_per_file[name] += n
    for (name, _), n in arrived.items():
        arrived_per_file[name] += n
    off = sorted(
        name
        for name in expected_per_file.keys() | arrived_per_file.keys()
        if expected_per_file[name] != arrived_per_file[name]
    )
    check(
        not off,
        "per-file row counts match the source",
        f"{len(off)} files differ: "
        + _fmt(
            f"{n}: got {arrived_per_file[n]}, want {expected_per_file[n]}" for n in off
        ),
    )

    print()
    if failures:
        print(f"FAILED {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All checks passed: every source row arrived exactly once.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sink", help="The sink main.py wrote to.")
    parser.add_argument(
        "--data-directory",
        default="300G-image-data-synthetic-raw-parquet",
        help=f"The source directory in the {SOURCE_BUCKET} bucket.",
    )
    args = parser.parse_args()
    return verify(args.sink, args.data_directory)


if __name__ == "__main__":
    sys.exit(main())
