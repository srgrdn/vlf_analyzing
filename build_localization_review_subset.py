#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


BUCKETS = ("empty", "single", "few", "dense")
MANIFEST_FIELDNAMES = [
    "sample_id",
    "channel",
    "bucket",
    "review_reason",
    "event_count",
    "source_file",
    "segment_start",
    "segment_end",
    "png_path",
    "npz_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a stratified manual-review subset from dataset_localization/metadata.csv. "
            "Copies review PNGs into per-channel bucket folders and writes a manifest."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to dataset_localization with metadata.csv and review/auto/<channel> PNGs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the review subset. Default: <dataset_dir>/review_subset",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--per-channel-total",
        type=int,
        default=80,
        help="Target total number of samples per channel. Default: 80",
    )
    parser.add_argument("--empty-count", type=int, default=20, help="Target empty samples per channel.")
    parser.add_argument("--single-count", type=int, default=20, help="Target single-event samples per channel.")
    parser.add_argument("--few-count", type=int, default=20, help="Target few-event (2-3) samples per channel.")
    parser.add_argument(
        "--dense-count",
        type=int,
        default=20,
        help="Target dense samples per channel, preferring the highest event_count.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to materialize selected PNGs. Default: copy",
    )
    return parser.parse_args()


def read_metadata(metadata_path: Path) -> list[dict[str, str]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with metadata_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ensure_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for channel in ("ns", "we"):
        for bucket in BUCKETS:
            (output_dir / channel / bucket).mkdir(parents=True, exist_ok=True)


def categorize_row(row: dict[str, str]) -> str:
    event_count = int(row["event_count"])
    if event_count == 0:
        return "empty"
    if event_count == 1:
        return "single"
    if event_count <= 3:
        return "few"
    return "dense"


def choose_rows(
    rows: list[dict[str, str]],
    channel: str,
    rng: random.Random,
    empty_count: int,
    single_count: int,
    few_count: int,
    dense_count: int,
    per_channel_total: int,
) -> list[tuple[dict[str, str], str]]:
    channel_rows = [row for row in rows if row["channel"] == channel]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in channel_rows:
        grouped[categorize_row(row)].append(row)

    selections: list[tuple[dict[str, str], str]] = []

    def pick_random(bucket: str, count: int, reason: str) -> None:
        candidates = grouped[bucket][:]
        rng.shuffle(candidates)
        for row in candidates[:count]:
            selections.append((row, reason))

    def pick_dense(count: int, reason: str) -> None:
        candidates = sorted(
            grouped["dense"],
            key=lambda row: (int(row["event_count"]), row["source_file"], row["segment_start"]),
            reverse=True,
        )
        for row in candidates[:count]:
            selections.append((row, reason))

    pick_random("empty", empty_count, "empty_reference")
    pick_random("single", single_count, "single_event_reference")
    pick_random("few", few_count, "few_events_reference")
    pick_dense(dense_count, "dense_events_priority")

    seen_ids = {row["sample_id"] for row, _ in selections}
    if len(selections) < per_channel_total:
        remaining = [row for row in channel_rows if row["sample_id"] not in seen_ids]
        remaining.sort(key=lambda row: (row["source_file"], row["segment_start"], row["sample_id"]))
        rng.shuffle(remaining)
        for row in remaining[: max(0, per_channel_total - len(selections))]:
            selections.append((row, "coverage_fill"))
    return selections


def materialize_png(source_path: Path, destination_path: Path, copy_mode: str) -> None:
    if copy_mode == "copy":
        shutil.copy2(source_path, destination_path)
        return
    if destination_path.exists():
        destination_path.unlink()
    destination_path.hardlink_to(source_path)


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else dataset_dir / "review_subset")
    metadata_path = dataset_dir / "metadata.csv"
    rows = read_metadata(metadata_path)
    ensure_output_dirs(output_dir)
    rng = random.Random(args.seed)

    manifest_rows: list[dict[str, str]] = []
    selected_total = 0

    for channel in ("ns", "we"):
        selections = choose_rows(
            rows=rows,
            channel=channel,
            rng=rng,
            empty_count=args.empty_count,
            single_count=args.single_count,
            few_count=args.few_count,
            dense_count=args.dense_count,
            per_channel_total=args.per_channel_total,
        )
        for row, reason in selections:
            png_source = dataset_dir / row["png_path"]
            npz_path = dataset_dir / row["npz_path"]
            bucket = categorize_row(row)
            destination = output_dir / channel / bucket / f"{row['sample_id']}.png"
            materialize_png(png_source, destination, args.copy_mode)
            manifest_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "channel": channel,
                    "bucket": bucket,
                    "review_reason": reason,
                    "event_count": row["event_count"],
                    "source_file": row["source_file"],
                    "segment_start": row["segment_start"],
                    "segment_end": row["segment_end"],
                    "png_path": str(destination.relative_to(output_dir)),
                    "npz_path": row["npz_path"],
                }
            )
        selected_total += len(selections)

    manifest_path = output_dir / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)

    print(f"Dataset dir   : {dataset_dir}")
    print(f"Output dir    : {output_dir}")
    print(f"Manifest      : {manifest_path}")
    print(f"Selected rows : {selected_total}")
    for channel in ("ns", "we"):
        count = sum(1 for row in manifest_rows if row["channel"] == channel)
        print(f"{channel:>2} samples   : {count}")


if __name__ == "__main__":
    main()
