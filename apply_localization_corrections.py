#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from build_localization_dataset import (
    DEFAULT_QUALITY_FLAG,
    METADATA_FIELDNAMES,
    build_target_heatmap,
)


ALLOWED_REVIEW_STATUSES = {
    "manual_verified",
    "manual_corrected",
    "artifact_suspected",
    "uncertain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply manual corrections from corrections.csv to dataset_localization metadata "
            "and NPZ samples, and materialize reviewed PNGs into verified/corrected folders."
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
        "--metadata-backup",
        type=Path,
        default=None,
        help="Optional backup path for metadata.csv before applying corrections.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate corrections and report planned changes without writing files.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def parse_event_times(raw: str | None, field_name: str, sample_id: str) -> np.ndarray:
    text = (raw or "").strip()
    if not text:
        return np.array([], dtype=np.float32)
    try:
        values = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} for sample_id={sample_id} is not valid JSON: {exc}") from exc
    if not isinstance(values, list):
        raise ValueError(f"{field_name} for sample_id={sample_id} must be a JSON list.")
    parsed = np.array([float(value) for value in values], dtype=np.float32)
    if np.any(parsed < 0):
        raise ValueError(f"{field_name} for sample_id={sample_id} contains negative times.")
    return np.sort(parsed)


def default_quality_flag(review_status: str) -> str:
    if review_status == "manual_verified":
        return "clean"
    if review_status == "manual_corrected":
        return "clean"
    if review_status == "artifact_suspected":
        return "artifact_suspected"
    if review_status == "uncertain":
        return "uncertain"
    return DEFAULT_QUALITY_FLAG


def destination_stage(review_status: str) -> str:
    if review_status == "manual_verified":
        return "verified"
    return "corrected"


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METADATA_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    corrections_path = (
        args.corrections.expanduser().resolve()
        if args.corrections
        else (dataset_dir / "review_subset" / "corrections.csv")
    )
    metadata_path = dataset_dir / "metadata.csv"
    metadata_rows = read_csv_rows(metadata_path)
    corrections_rows = read_csv_rows(corrections_path)
    metadata_by_id = {row["sample_id"]: row for row in metadata_rows}

    updates: list[tuple[str, np.ndarray, str, str, Path]] = []

    for correction in corrections_rows:
        review_status = correction["review_status"].strip()
        if not review_status:
            continue
        if review_status not in ALLOWED_REVIEW_STATUSES:
            raise ValueError(
                f"sample_id={correction['sample_id']} has unsupported review_status={review_status!r}"
            )

        sample_id = correction["sample_id"]
        metadata_row = metadata_by_id.get(sample_id)
        if metadata_row is None:
            raise KeyError(f"sample_id {sample_id} from corrections is missing in metadata.csv")

        auto_event_times = parse_event_times(
            metadata_row.get("event_times_s_json", ""),
            field_name="event_times_s_json",
            sample_id=sample_id,
        )
        corrected_raw = correction.get("corrected_event_times_s_json", "")
        if corrected_raw.strip():
            final_event_times = parse_event_times(
                corrected_raw,
                field_name="corrected_event_times_s_json",
                sample_id=sample_id,
            )
        else:
            final_event_times = auto_event_times

        quality_flag = correction["quality_flag"].strip() or default_quality_flag(review_status)
        npz_path = dataset_dir / metadata_row["npz_path"]
        updates.append((sample_id, final_event_times, review_status, quality_flag, npz_path))

    if not updates:
        print("No corrections with review_status were found. Nothing to apply.")
        return

    for sample_id, final_event_times, review_status, quality_flag, npz_path in updates:
        npz = np.load(npz_path, allow_pickle=True)
        time_axis = np.asarray(npz["time_axis"], dtype=np.float32)
        target_heatmap = build_target_heatmap(time_axis, final_event_times, sigma_s=0.01)

        metadata_row = metadata_by_id[sample_id]
        metadata_row["event_count"] = str(int(final_event_times.size))
        metadata_row["event_times_s_json"] = json.dumps([float(value) for value in final_event_times])
        metadata_row["label_source"] = review_status
        metadata_row["quality_flag"] = quality_flag

        channel = metadata_row["channel"]
        png_auto_path = dataset_dir / metadata_row["png_path"]
        destination_dir = dataset_dir / "review" / destination_stage(review_status) / channel
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_png = destination_dir / f"{sample_id}.png"

        if args.dry_run:
            continue

        np.savez_compressed(
            npz_path,
            spec_db=np.asarray(npz["spec_db"], dtype=np.float32),
            envelope_db=np.asarray(npz["envelope_db"], dtype=np.float32),
            time_axis=time_axis.astype(np.float32),
            freq_axis=np.asarray(npz["freq_axis"], dtype=np.float32),
            source_file=np.asarray(npz["source_file"]),
            channel=np.asarray(npz["channel"]),
            segment_start=np.asarray(npz["segment_start"], dtype=np.float32),
            segment_end=np.asarray(npz["segment_end"], dtype=np.float32),
            event_times_s=final_event_times.astype(np.float32),
            event_count=np.array(final_event_times.size, dtype=np.int32),
            target_heatmap=target_heatmap.astype(np.float32),
            label_source=np.array(review_status),
            quality_flag=np.array(quality_flag),
        )

        shutil.copy2(png_auto_path, destination_png)
        other_stage = "corrected" if destination_stage(review_status) == "verified" else "verified"
        other_png = dataset_dir / "review" / other_stage / channel / f"{sample_id}.png"
        if other_png.exists():
            other_png.unlink()

    if not args.dry_run:
        backup_path = (
            args.metadata_backup.expanduser().resolve()
            if args.metadata_backup
            else metadata_path.with_name("metadata.pre_review_backup.csv")
        )
        if not backup_path.exists():
            shutil.copy2(metadata_path, backup_path)
        write_metadata(metadata_path, metadata_rows)

    print(f"Dataset dir     : {dataset_dir}")
    print(f"Corrections     : {corrections_path}")
    print(f"Applied entries : {len(updates)}")
    print(f"Dry run         : {args.dry_run}")


if __name__ == "__main__":
    main()
