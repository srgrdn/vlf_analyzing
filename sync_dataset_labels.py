#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


LABEL_DIRS = ("burst", "no_burst")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync manual labels from dataset/review/{burst,no_burst} into metadata.csv."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"), help="Dataset directory.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create metadata.csv.bak.")
    return parser.parse_args()


def collect_labels(dataset_dir: Path) -> dict[str, tuple[str, str]]:
    labels: dict[str, tuple[str, str]] = {}
    review_dir = dataset_dir / "review"
    for label in LABEL_DIRS:
        label_dir = review_dir / label
        for path in sorted(label_dir.glob("*.png")):
            sample_id = path.stem
            labels[sample_id] = (label, str(path.relative_to(dataset_dir)))
    return labels


def assign_splits(rows: list[dict[str, str]], seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.0.")

    def key(row: dict[str, str]) -> int:
        digest = hashlib.sha256(f"{seed}:{row['sample_id']}".encode("utf-8")).hexdigest()
        return int(digest, 16)

    for label in LABEL_DIRS:
        labeled_rows = [row for row in rows if row.get("label") == label]
        labeled_rows.sort(key=key)
        n = len(labeled_rows)
        train_end = int(round(n * train_ratio))
        val_end = train_end + int(round(n * val_ratio))
        for idx, row in enumerate(labeled_rows):
            if idx < train_end:
                row["split"] = "train"
            elif idx < val_end:
                row["split"] = "val"
            else:
                row["split"] = "test"


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        raise ValueError(f"Metadata file does not exist: {metadata_path}")

    labels = collect_labels(dataset_dir)
    with metadata_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if fieldnames is None:
        raise ValueError("metadata.csv has no header.")

    updated = 0
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id not in labels:
            continue
        label, png_path = labels[sample_id]
        row["label"] = label
        row["png_path"] = png_path
        row["label_source"] = "manual"
        updated += 1

    assign_splits(rows, args.seed, args.train_ratio, args.val_ratio, args.test_ratio)

    if not args.no_backup:
        shutil.copy2(metadata_path, metadata_path.with_suffix(".csv.bak"))

    with metadata_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {label: sum(row.get("label") == label for row in rows) for label in LABEL_DIRS}
    split_counts: dict[str, dict[str, int]] = {}
    for label in LABEL_DIRS:
        split_counts[label] = {
            split: sum(row.get("label") == label and row.get("split") == split for row in rows)
            for split in ("train", "val", "test")
        }

    print(f"Metadata        : {metadata_path}")
    print(f"Manual labels   : {updated}")
    for label in LABEL_DIRS:
        print(
            f"{label:8}       : {counts[label]} "
            f"(train={split_counts[label]['train']}, "
            f"val={split_counts[label]['val']}, "
            f"test={split_counts[label]['test']})"
        )


if __name__ == "__main__":
    main()
