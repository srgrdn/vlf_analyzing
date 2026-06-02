#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CORRECTIONS_FIELDNAMES = [
    "sample_id",
    "channel",
    "bucket",
    "source_file",
    "segment_start",
    "segment_end",
    "auto_event_count",
    "auto_event_times_s_json",
    "review_status",
    "corrected_event_times_s_json",
    "quality_flag",
    "comment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a corrections.csv template for manual review from "
            "dataset_localization/review_subset/manifest.csv."
        )
    )
    parser.add_argument(
        "review_subset_dir",
        type=Path,
        help="Path to review_subset with manifest.csv.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Path to dataset_localization. Default: parent of review_subset_dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output corrections CSV. Default: <review_subset_dir>/corrections.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing corrections CSV.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    args = parse_args()
    review_subset_dir = args.review_subset_dir.expanduser().resolve()
    dataset_dir = (args.dataset_dir.expanduser().resolve() if args.dataset_dir else review_subset_dir.parent)
    output_path = (args.output.expanduser().resolve() if args.output else review_subset_dir / "corrections.csv")
    manifest_path = review_subset_dir / "manifest.csv"
    metadata_path = dataset_dir / "metadata.csv"

    if output_path.exists() and not args.overwrite:
        raise ValueError(f"{output_path} already exists. Use --overwrite to rebuild it.")

    manifest_rows = read_csv_rows(manifest_path)
    metadata_rows = read_csv_rows(metadata_path)
    metadata_by_id = {row["sample_id"]: row for row in metadata_rows}

    correction_rows: list[dict[str, str]] = []
    for manifest_row in manifest_rows:
        sample_id = manifest_row["sample_id"]
        metadata_row = metadata_by_id.get(sample_id)
        if metadata_row is None:
            raise KeyError(f"sample_id {sample_id} from manifest is missing in metadata.csv")
        correction_rows.append(
            {
                "sample_id": sample_id,
                "channel": manifest_row["channel"],
                "bucket": manifest_row["bucket"],
                "source_file": manifest_row["source_file"],
                "segment_start": manifest_row["segment_start"],
                "segment_end": manifest_row["segment_end"],
                "auto_event_count": metadata_row["event_count"],
                "auto_event_times_s_json": metadata_row["event_times_s_json"] or json.dumps([]),
                "review_status": "",
                "corrected_event_times_s_json": "",
                "quality_flag": "",
                "comment": "",
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CORRECTIONS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(correction_rows)

    print(f"Review subset : {review_subset_dir}")
    print(f"Dataset dir   : {dataset_dir}")
    print(f"Corrections   : {output_path}")
    print(f"Rows          : {len(correction_rows)}")


if __name__ == "__main__":
    main()
