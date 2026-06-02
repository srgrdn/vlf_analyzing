#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


KNOWN_REVIEW_STATUSES = {
    "manual_verified",
    "manual_corrected",
    "artifact_suspected",
    "uncertain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize manual review progress for "
            "dataset_localization/review_subset/corrections.csv without applying changes."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to dataset_localization.",
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        default=None,
        help="Path to corrections.csv. Default: <dataset_dir>/review_subset/corrections.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of pending examples to print. Default: 10",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def clean(value: str | None) -> str:
    return (value or "").strip()


def status_label(status: str) -> str:
    return status if status else "pending"


def print_counter(title: str, counter: Counter[str]) -> None:
    print(title)
    if not counter:
        print("  (none)")
        return
    for key in sorted(counter):
        print(f"  {key}: {counter[key]}")


def validate_rows(manifest_rows: list[dict[str, str]], correction_rows: list[dict[str, str]]) -> None:
    manifest_ids = {row["sample_id"] for row in manifest_rows}
    correction_ids = [row["sample_id"] for row in correction_rows]
    duplicate_ids = sorted(sample_id for sample_id, count in Counter(correction_ids).items() if count > 1)
    unknown_ids = sorted(set(correction_ids) - manifest_ids)
    missing_ids = sorted(manifest_ids - set(correction_ids))

    if duplicate_ids:
        raise ValueError(f"Duplicate sample_id values in corrections.csv: {', '.join(duplicate_ids[:10])}")
    if unknown_ids:
        raise ValueError(f"Corrections contain sample_id values missing from manifest.csv: {', '.join(unknown_ids[:10])}")
    if missing_ids:
        raise ValueError(f"Corrections are missing manifest sample_id values: {', '.join(missing_ids[:10])}")


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")

    dataset_dir = args.dataset_dir.expanduser().resolve()
    review_subset_dir = dataset_dir / "review_subset"
    manifest_path = review_subset_dir / "manifest.csv"
    corrections_path = (
        args.corrections.expanduser().resolve()
        if args.corrections
        else review_subset_dir / "corrections.csv"
    )

    manifest_rows = read_csv_rows(manifest_path)
    correction_rows = read_csv_rows(corrections_path)
    validate_rows(manifest_rows, correction_rows)

    manifest_by_id = {row["sample_id"]: row for row in manifest_rows}
    total = len(correction_rows)
    reviewed_rows = [row for row in correction_rows if clean(row.get("review_status"))]
    pending_rows = [row for row in correction_rows if not clean(row.get("review_status"))]
    reviewed_count = len(reviewed_rows)
    pending_count = len(pending_rows)
    complete_pct = (reviewed_count / total * 100.0) if total else 0.0

    status_counts: Counter[str] = Counter()
    channel_counts: dict[str, Counter[str]] = defaultdict(Counter)
    bucket_counts: dict[str, Counter[str]] = defaultdict(Counter)
    quality_counts: Counter[str] = Counter()
    unknown_statuses: Counter[str] = Counter()

    for correction in correction_rows:
        sample_id = correction["sample_id"]
        manifest_row = manifest_by_id[sample_id]
        status = clean(correction.get("review_status"))
        label = status_label(status)
        status_counts[label] += 1
        channel_counts[manifest_row["channel"]][label] += 1
        bucket_counts[manifest_row["bucket"]][label] += 1

        quality_flag = clean(correction.get("quality_flag"))
        quality_counts[quality_flag if quality_flag else "(blank)"] += 1
        if status and status not in KNOWN_REVIEW_STATUSES:
            unknown_statuses[status] += 1

    print(f"Dataset dir   : {dataset_dir}")
    print(f"Manifest      : {manifest_path}")
    print(f"Corrections   : {corrections_path}")
    print(f"Total rows    : {total}")
    print(f"Reviewed      : {reviewed_count}")
    print(f"Pending       : {pending_count}")
    print(f"Complete      : {complete_pct:.1f}%")
    print()

    print_counter("By review_status:", status_counts)
    print()

    print("By channel:")
    for channel in sorted(channel_counts):
        parts = ", ".join(f"{key}={channel_counts[channel][key]}" for key in sorted(channel_counts[channel]))
        print(f"  {channel}: {parts}")
    print()

    print("By bucket:")
    for bucket in sorted(bucket_counts):
        parts = ", ".join(f"{key}={bucket_counts[bucket][key]}" for key in sorted(bucket_counts[bucket]))
        print(f"  {bucket}: {parts}")
    print()

    print_counter("By quality_flag:", quality_counts)
    if unknown_statuses:
        print()
        print_counter("Unknown review_status values:", unknown_statuses)

    if args.limit == 0:
        return

    print()
    print(f"Next pending examples (limit={args.limit}):")
    if not pending_rows:
        print("  (none)")
        return

    for correction in pending_rows[: args.limit]:
        sample_id = correction["sample_id"]
        manifest_row = manifest_by_id[sample_id]
        png_path = review_subset_dir / manifest_row["png_path"]
        print(
            "  "
            f"{sample_id} | "
            f"channel={manifest_row['channel']} | "
            f"bucket={manifest_row['bucket']} | "
            f"events={manifest_row['event_count']} | "
            f"segment={manifest_row['segment_start']}-{manifest_row['segment_end']}s | "
            f"png={png_path}"
        )


if __name__ == "__main__":
    main()
