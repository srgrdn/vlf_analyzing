#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from build_localization_dataset import METADATA_FIELDNAMES


REVIEWED_LABEL_SOURCES = ("manual_verified", "manual_corrected")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assign train/val/test splits to reviewed localization samples while "
            "keeping all reviewed rows from the same source_file in the same split."
        )
    )
    parser.add_argument("dataset_dir", type=Path, help="Path to dataset_localization.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for source_file shuffling.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Target train ratio. Default: 0.70")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Target validation ratio. Default: 0.15")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Target test ratio. Default: 0.15")
    parser.add_argument(
        "--label-source",
        action="append",
        choices=REVIEWED_LABEL_SOURCES,
        default=None,
        help=(
            "Reviewed label_source to include. Can be repeated. "
            "Default: manual_verified and manual_corrected."
        ),
    )
    parser.add_argument(
        "--metadata-backup",
        type=Path,
        default=None,
        help="Optional backup path for metadata.csv before writing splits.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report planned split assignments without writing.")
    return parser.parse_args()


def read_metadata(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METADATA_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("At least one split ratio must be positive.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}.")


def choose_split(counts: Counter[str], group_size: int, targets: dict[str, float]) -> str:
    best_split = "train"
    best_score: tuple[float, int, str] | None = None
    for split in SPLITS:
        if targets[split] <= 0:
            continue
        projected = counts[split] + group_size
        score = (projected - targets[split], counts[split], split)
        if best_score is None or score < best_score:
            best_score = score
            best_split = split
    return best_split


def build_source_file_assignments(
    reviewed_rows: list[dict[str, str]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in reviewed_rows:
        grouped[row["source_file"]].append(row)

    source_files = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(source_files)

    total_rows = len(reviewed_rows)
    targets = {
        "train": total_rows * train_ratio,
        "val": total_rows * val_ratio,
        "test": total_rows * test_ratio,
    }
    counts: Counter[str] = Counter()
    assignments: dict[str, str] = {}

    for source_file in source_files:
        group_size = len(grouped[source_file])
        split = choose_split(counts, group_size, targets)
        assignments[source_file] = split
        counts[split] += group_size

    return assignments


def print_summary(rows: list[dict[str, str]], reviewed_rows: list[dict[str, str]], assignments: dict[str, str]) -> None:
    split_counts: Counter[str] = Counter()
    split_channel_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_source_counts: dict[str, set[str]] = defaultdict(set)

    for row in reviewed_rows:
        split = assignments[row["source_file"]]
        split_counts[split] += 1
        split_channel_counts[split][row["channel"]] += 1
        split_label_counts[split][row["label_source"]] += 1
        split_source_counts[split].add(row["source_file"])

    print(f"Metadata rows        : {len(rows)}")
    print(f"Reviewed rows        : {len(reviewed_rows)}")
    print(f"Reviewed source files: {len(assignments)}")
    print()
    print("Split rows:")
    for split in SPLITS:
        print(f"  {split}: {split_counts[split]}")
    print()
    print("Split source_files:")
    for split in SPLITS:
        print(f"  {split}: {len(split_source_counts[split])}")
    print()
    print("Split channels:")
    for split in SPLITS:
        parts = ", ".join(f"{key}={split_channel_counts[split][key]}" for key in sorted(split_channel_counts[split]))
        print(f"  {split}: {parts or '(none)'}")
    print()
    print("Split label_source:")
    for split in SPLITS:
        parts = ", ".join(f"{key}={split_label_counts[split][key]}" for key in sorted(split_label_counts[split]))
        print(f"  {split}: {parts or '(none)'}")


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    dataset_dir = args.dataset_dir.expanduser().resolve()
    metadata_path = dataset_dir / "metadata.csv"
    metadata_rows = read_metadata(metadata_path)
    included_label_sources = set(args.label_source or REVIEWED_LABEL_SOURCES)
    reviewed_rows = [row for row in metadata_rows if row["label_source"] in included_label_sources]

    if not reviewed_rows:
        raise ValueError(f"No reviewed rows found with label_source in {sorted(included_label_sources)}.")

    assignments = build_source_file_assignments(
        reviewed_rows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print_summary(metadata_rows, reviewed_rows, assignments)
    print()
    print(f"Dry run: {args.dry_run}")

    if args.dry_run:
        return

    for row in metadata_rows:
        if row["label_source"] in included_label_sources:
            row["split"] = assignments[row["source_file"]]
        else:
            row["split"] = ""

    backup_path = (
        args.metadata_backup.expanduser().resolve()
        if args.metadata_backup
        else metadata_path.with_name("metadata.pre_split_backup.csv")
    )
    if not backup_path.exists():
        shutil.copy2(metadata_path, backup_path)
    write_metadata(metadata_path, metadata_rows)
    print(f"Metadata updated: {metadata_path}")
    print(f"Backup          : {backup_path}")


if __name__ == "__main__":
    main()
