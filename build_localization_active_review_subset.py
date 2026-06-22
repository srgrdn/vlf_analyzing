#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from build_localization_review_subset import BUCKETS, MANIFEST_FIELDNAMES, categorize_row, materialize_png


REVIEWED_LABEL_SOURCES = {"manual_verified", "manual_corrected"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a second-pass active-review subset from unreviewed localization rows. "
            "The selection prioritizes model-error source files when prediction reports are "
            "available, then fills channel/bucket quotas for broader reviewed coverage."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to dataset_localization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <dataset_dir>/review_subset_v2",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        action="append",
        default=None,
        help=(
            "Optional model prediction CSV used to prioritize source_file/channel combinations. "
            "Can be repeated. Defaults to 2D localization val/test prediction CSVs if present."
        ),
    )
    parser.add_argument("--seed", type=int, default=43, help="Random seed. Default: 43")
    parser.add_argument("--per-channel-total", type=int, default=200, help="Target samples per channel. Default: 200")
    parser.add_argument("--empty-count", type=int, default=35, help="Target empty samples per channel. Default: 35")
    parser.add_argument("--single-count", type=int, default=45, help="Target single-event samples per channel. Default: 45")
    parser.add_argument("--few-count", type=int, default=55, help="Target few-event samples per channel. Default: 55")
    parser.add_argument("--dense-count", type=int, default=65, help="Target dense samples per channel. Default: 65")
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to materialize selected PNGs. Default: copy",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dirs(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite and any(output_dir.iterdir()):
            raise ValueError(f"{output_dir} is not empty. Use --overwrite to rebuild it.")
        if overwrite:
            shutil.rmtree(output_dir)
    for channel in ("ns", "we"):
        for bucket in BUCKETS:
            (output_dir / channel / bucket).mkdir(parents=True, exist_ok=True)


def default_prediction_paths(dataset_dir: Path) -> list[Path]:
    project_root = dataset_dir.parent
    reports_dir = project_root / "methods" / "2d_cnn_localization" / "reports"
    return [
        reports_dir / "sferics_localization_2d_val_predictions.csv",
        reports_dir / "sferics_localization_2d_test_predictions.csv",
    ]


def build_error_priority(prediction_paths: list[Path]) -> dict[tuple[str, str], float]:
    priority: dict[tuple[str, str], float] = defaultdict(float)
    for path in prediction_paths:
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            source_file = row.get("source_file", "")
            channel = row.get("channel", "")
            if not source_file or not channel:
                continue
            event_fp = int(float(row.get("event_fp") or 0))
            event_fn = int(float(row.get("event_fn") or 0))
            true_count = int(float(row.get("event_count") or 0))
            predicted_count = int(float(row.get("predicted_event_count") or 0))
            heatmap_mse = float(row.get("heatmap_mse") or 0.0)
            score = event_fp + event_fn + 0.5 * abs(predicted_count - true_count) + 10.0 * heatmap_mse
            if score > 0:
                priority[(source_file, channel)] += score
    return dict(priority)


def row_priority(row: dict[str, str], error_priority: dict[tuple[str, str], float]) -> float:
    bucket = categorize_row(row)
    event_count = int(row["event_count"])
    score = error_priority.get((row["source_file"], row["channel"]), 0.0)
    if row["channel"] == "we":
        score += 0.75
    if bucket == "dense":
        score += min(event_count, 12) * 0.25
    elif bucket == "few":
        score += 1.0
    elif bucket == "single":
        score += 0.4
    else:
        score += 0.25
    return score


def choose_bucket_rows(
    candidates: list[dict[str, str]],
    *,
    count: int,
    rng: random.Random,
    error_priority: dict[tuple[str, str], float],
) -> list[tuple[dict[str, str], str]]:
    randomized = [(rng.random(), row) for row in candidates]
    randomized.sort(
        key=lambda item: (
            row_priority(item[1], error_priority),
            item[1]["source_file"],
            float(item[1]["segment_start"]),
            item[0],
        ),
        reverse=True,
    )
    selections: list[tuple[dict[str, str], str]] = []
    for _, row in randomized[:count]:
        base_reason = f"active_{categorize_row(row)}"
        if (row["source_file"], row["channel"]) in error_priority:
            base_reason += "_model_error_source"
        selections.append((row, base_reason))
    return selections


def choose_rows(
    rows: list[dict[str, str]],
    channel: str,
    rng: random.Random,
    error_priority: dict[tuple[str, str], float],
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
    for bucket, count in (
        ("empty", empty_count),
        ("single", single_count),
        ("few", few_count),
        ("dense", dense_count),
    ):
        selections.extend(
            choose_bucket_rows(
                grouped[bucket],
                count=count,
                rng=rng,
                error_priority=error_priority,
            )
        )

    seen_ids = {row["sample_id"] for row, _ in selections}
    if len(selections) < per_channel_total:
        remaining = [row for row in channel_rows if row["sample_id"] not in seen_ids]
        fill_rows = choose_bucket_rows(
            remaining,
            count=per_channel_total - len(selections),
            rng=rng,
            error_priority=error_priority,
        )
        selections.extend((row, reason.replace("active_", "active_coverage_fill_")) for row, reason in fill_rows)
    return selections


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else dataset_dir / "review_subset_v2"
    metadata_rows = read_csv_rows(dataset_dir / "metadata.csv")
    prediction_paths = [path.expanduser().resolve() for path in args.predictions] if args.predictions else default_prediction_paths(dataset_dir)
    error_priority = build_error_priority(prediction_paths)
    prepare_output_dirs(output_dir, overwrite=args.overwrite)
    rng = random.Random(args.seed)

    candidates = [
        row
        for row in metadata_rows
        if row["label_source"] not in REVIEWED_LABEL_SOURCES
        and not row.get("split", "").strip()
    ]
    if not candidates:
        raise ValueError("No unreviewed localization rows are available for active review.")

    manifest_rows: list[dict[str, str]] = []
    selected_ids: set[str] = set()
    for channel in ("ns", "we"):
        selections = choose_rows(
            candidates,
            channel=channel,
            rng=rng,
            error_priority=error_priority,
            empty_count=args.empty_count,
            single_count=args.single_count,
            few_count=args.few_count,
            dense_count=args.dense_count,
            per_channel_total=args.per_channel_total,
        )
        for row, reason in selections:
            if row["sample_id"] in selected_ids:
                continue
            selected_ids.add(row["sample_id"])
            bucket = categorize_row(row)
            png_source = dataset_dir / row["png_path"]
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

    manifest_path = output_dir / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)

    print(f"Dataset dir       : {dataset_dir}")
    print(f"Output dir        : {output_dir}")
    print(f"Manifest          : {manifest_path}")
    print(f"Prediction reports: {sum(1 for path in prediction_paths if path.exists())}/{len(prediction_paths)}")
    print(f"Error priorities  : {len(error_priority)} source_file/channel pairs")
    print(f"Candidate rows    : {len(candidates)}")
    print(f"Selected rows     : {len(manifest_rows)}")
    print(f"By channel        : {dict(Counter(row['channel'] for row in manifest_rows))}")
    print(f"By bucket         : {dict(Counter(row['bucket'] for row in manifest_rows))}")
    print(f"By reason         : {dict(Counter(row['review_reason'] for row in manifest_rows))}")


if __name__ == "__main__":
    main()
