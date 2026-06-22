from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np


REVIEWED_LABEL_SOURCES = {"manual_verified", "manual_corrected"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a baseline 2D CNN that maps localization spectrograms "
            "(`spec_db`) to temporal event heatmaps (`target_heatmap`)."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset_localization"),
        help="Path to dataset_localization. Default: dataset_localization",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: directory containing this script.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs. Default: 40")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size. Default: 16")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate. Default: 1e-3")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay. Default: 1e-4")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. Default: 0")
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=0.35,
        help="Threshold for simple predicted peak extraction in reports. Default: 0.35",
    )
    parser.add_argument(
        "--min-peak-distance",
        type=float,
        default=0.03,
        help="Minimum predicted peak distance in seconds for reports. Default: 0.03",
    )
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=0.04,
        help="Event-time matching tolerance in seconds for reports. Default: 0.04",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_reviewed_metadata(dataset_dir: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(dataset_dir / "metadata.csv")
    reviewed = [
        row
        for row in rows
        if row["label_source"] in REVIEWED_LABEL_SOURCES
        and row["quality_flag"] == "clean"
        and row["split"] in SPLITS
    ]
    if not reviewed:
        raise ValueError(
            "No reviewed localization rows found. Expected manual_verified/manual_corrected "
            "rows with quality_flag=clean and non-empty split."
        )
    return reviewed


def rows_by_split(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    frames = {split: [row for row in rows if row["split"] == split] for split in SPLITS}
    missing = [split for split, split_rows in frames.items() if not split_rows]
    if missing:
        raise ValueError(f"Missing reviewed rows for split(s): {', '.join(missing)}")
    return frames


def normalize_spec(spec_db: np.ndarray) -> np.ndarray:
    spec = np.asarray(spec_db, dtype=np.float32)
    return (spec - float(spec.mean())) / (float(spec.std()) + 1e-6)


def pick_peaks(series: np.ndarray, threshold: float, min_distance_bins: int) -> np.ndarray:
    values = np.asarray(series, dtype=np.float32)
    if values.size < 3:
        return np.array([], dtype=np.int64)
    candidates = np.flatnonzero(
        (values[1:-1] >= values[:-2])
        & (values[1:-1] > values[2:])
        & (values[1:-1] >= threshold)
    ) + 1
    if candidates.size == 0:
        return candidates.astype(np.int64)

    order = candidates[np.argsort(values[candidates])[::-1]]
    selected: list[int] = []
    for index in order:
        if all(abs(int(index) - prev) >= min_distance_bins for prev in selected):
            selected.append(int(index))
    return np.array(sorted(selected), dtype=np.int64)


def parse_event_times(raw: str | None) -> list[float]:
    if not raw:
        return []
    values = json.loads(raw)
    if not isinstance(values, list):
        raise ValueError("event_times_s_json must contain a JSON list.")
    return sorted(float(value) for value in values)


def match_event_times(true_times: list[float], predicted_times: list[float], tolerance_s: float) -> tuple[int, int, int]:
    matched_true: set[int] = set()
    true_positive = 0
    for predicted_time in predicted_times:
        best_index = None
        best_distance = None
        for index, true_time in enumerate(true_times):
            if index in matched_true:
                continue
            distance = abs(predicted_time - true_time)
            if distance <= tolerance_s and (best_distance is None or distance < best_distance):
                best_index = index
                best_distance = distance
        if best_index is not None:
            matched_true.add(best_index)
            true_positive += 1
    false_positive = len(predicted_times) - true_positive
    false_negative = len(true_times) - true_positive
    return true_positive, false_positive, false_negative


def summarize_event_matches(rows: list[dict[str, object]]) -> dict[str, float]:
    true_positive = sum(int(row["event_tp"]) for row in rows)
    false_positive = sum(int(row["event_fp"]) for row in rows)
    false_negative = sum(int(row["event_fn"]) for row in rows)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    count_mae = float(np.mean([abs(int(row["predicted_event_count"]) - int(row["event_count"])) for row in rows])) if rows else 0.0
    return {
        "event_tp": float(true_positive),
        "event_fp": float(false_positive),
        "event_fn": float(false_negative),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "event_count_mae": count_mae,
    }


def import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for training. Install torch in the active environment "
            "before running this script."
        ) from exc
    return torch, nn, DataLoader, Dataset


def build_torch_components():
    torch, nn, DataLoader, Dataset = import_torch()

    class LocalizationHeatmapDataset(Dataset):
        def __init__(self, rows: list[dict[str, str]], dataset_dir: Path):
            self.rows = list(rows)
            self.dataset_dir = dataset_dir

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            npz_path = self.dataset_dir / Path(str(row["npz_path"]).replace("\\", "/"))
            data = np.load(npz_path)
            spec = normalize_spec(data["spec_db"])
            target = np.asarray(data["target_heatmap"], dtype=np.float32)
            return torch.from_numpy(spec[None, :, :]), torch.from_numpy(target)

    class SfericsLocalizationCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
            self.temporal_head = nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Conv1d(64, 1, kernel_size=1),
            )

        def forward(self, x):
            features = self.features(x)
            temporal = features.mean(dim=2)
            logits = self.temporal_head(temporal).squeeze(1)
            return torch.sigmoid(logits)

    return torch, nn, DataLoader, LocalizationHeatmapDataset, SfericsLocalizationCNN


def build_loaders(rows: dict[str, list[dict[str, str]]], dataset_dir: Path, batch_size: int, num_workers: int):
    _, _, DataLoader, LocalizationHeatmapDataset, _ = build_torch_components()
    return {
        "train": DataLoader(
            LocalizationHeatmapDataset(rows["train"], dataset_dir),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            LocalizationHeatmapDataset(rows["val"], dataset_dir),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            LocalizationHeatmapDataset(rows["test"], dataset_dir),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0
    torch = import_torch()[0]

    for spec, target in loader:
        spec = spec.to(device)
        target = target.to(device)
        with torch.set_grad_enabled(train):
            prediction = model(spec)
            loss = criterion(prediction, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = int(spec.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_mae += float(torch.mean(torch.abs(prediction.detach() - target)).item()) * batch_size
        total_count += batch_size

    return {
        "loss": total_loss / total_count,
        "mae": total_mae / total_count,
    }


def predict_split(
    model,
    rows: list[dict[str, str]],
    dataset_dir: Path,
    device,
    peak_threshold: float,
    min_peak_distance_s: float,
    match_tolerance_s: float,
):
    torch = import_torch()[0]
    model.eval()
    output_rows: list[dict[str, object]] = []
    mse_values: list[float] = []
    mae_values: list[float] = []

    with torch.no_grad():
        for row in rows:
            npz_path = dataset_dir / Path(str(row["npz_path"]).replace("\\", "/"))
            data = np.load(npz_path)
            spec = normalize_spec(data["spec_db"])
            target = np.asarray(data["target_heatmap"], dtype=np.float32)
            time_axis = np.asarray(data["time_axis"], dtype=np.float32)
            tensor = torch.from_numpy(spec[None, None, :, :]).to(device)
            prediction = model(tensor).detach().cpu().numpy()[0].astype(np.float32)
            mse = float(np.mean((prediction - target) ** 2))
            mae = float(np.mean(np.abs(prediction - target)))
            mse_values.append(mse)
            mae_values.append(mae)

            time_step = float(np.median(np.diff(time_axis))) if time_axis.size > 1 else 0.002
            min_distance_bins = max(1, int(round(min_peak_distance_s / time_step)))
            peak_indices = pick_peaks(prediction, threshold=peak_threshold, min_distance_bins=min_distance_bins)
            predicted_times = [float(time_axis[index]) for index in peak_indices]
            true_times = parse_event_times(row["event_times_s_json"])
            event_tp, event_fp, event_fn = match_event_times(
                true_times=true_times,
                predicted_times=predicted_times,
                tolerance_s=match_tolerance_s,
            )

            output_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "source_file": row["source_file"],
                    "channel": row["channel"],
                    "segment_start": row["segment_start"],
                    "segment_end": row["segment_end"],
                    "label_source": row["label_source"],
                    "event_count": row["event_count"],
                    "event_times_s_json": row["event_times_s_json"],
                    "predicted_event_count": len(predicted_times),
                    "predicted_event_times_s_json": json.dumps(predicted_times),
                    "event_tp": event_tp,
                    "event_fp": event_fp,
                    "event_fn": event_fn,
                    "heatmap_mse": mse,
                    "heatmap_mae": mae,
                    "npz_path": row["npz_path"],
                    "png_path": row["png_path"],
                }
            )

    return output_rows, {
        "heatmap_mse": float(np.mean(mse_values)) if mse_values else 0.0,
        "heatmap_mae": float(np.mean(mae_values)) if mae_values else 0.0,
        **summarize_event_matches(output_rows),
    }


def train(args: argparse.Namespace) -> None:
    torch, nn, _, _, SfericsLocalizationCNN = build_torch_components()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    method_dir = Path(__file__).resolve().parent
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else method_dir)
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(6, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = load_reviewed_metadata(dataset_dir)
    split_rows = rows_by_split(metadata)
    loaders = build_loaders(split_rows, dataset_dir, args.batch_size, args.num_workers)

    model = SfericsLocalizationCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_state = None
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, loaders["val"], criterion, optimizer, device, train=False)
        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
        }
        history.append(history_row)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch {epoch:02d}: "
            f"train_loss={train_metrics['loss']:.6f} train_mae={train_metrics['mae']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_mae={val_metrics['mae']:.6f}"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")
    model.load_state_dict(best_state)

    val_predictions, val_report = predict_split(
        model,
        split_rows["val"],
        dataset_dir,
        device,
        peak_threshold=args.peak_threshold,
        min_peak_distance_s=args.min_peak_distance,
        match_tolerance_s=args.match_tolerance,
    )
    test_predictions, test_report = predict_split(
        model,
        split_rows["test"],
        dataset_dir,
        device,
        peak_threshold=args.peak_threshold,
        min_peak_distance_s=args.min_peak_distance,
        match_tolerance_s=args.match_tolerance,
    )

    report = {
        "model_kind": "sferics_localization_2d",
        "dataset_dir": str(dataset_dir),
        "train_samples": len(split_rows["train"]),
        "val_samples": len(split_rows["val"]),
        "test_samples": len(split_rows["test"]),
        "best_val_loss": float(best_val_loss),
        "peak_threshold": args.peak_threshold,
        "min_peak_distance": args.min_peak_distance,
        "match_tolerance": args.match_tolerance,
        "val": val_report,
        "test": test_report,
        "history": history,
    }

    model_path = models_dir / "sferics_localization_2d.pt"
    metrics_path = reports_dir / "sferics_localization_2d_metrics.json"
    val_predictions_path = reports_dir / "sferics_localization_2d_val_predictions.csv"
    test_predictions_path = reports_dir / "sferics_localization_2d_test_predictions.csv"

    torch.save(
        {
            "model_kind": "sferics_localization_2d",
            "model_state_dict": model.state_dict(),
            "input_key": "spec_db",
            "target_key": "target_heatmap",
            "reviewed_label_sources": sorted(REVIEWED_LABEL_SOURCES),
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    prediction_fields = [
        "sample_id",
        "source_file",
        "channel",
        "segment_start",
        "segment_end",
        "label_source",
        "event_count",
        "event_times_s_json",
        "predicted_event_count",
        "predicted_event_times_s_json",
        "event_tp",
        "event_fp",
        "event_fn",
        "heatmap_mse",
        "heatmap_mae",
        "npz_path",
        "png_path",
    ]
    write_csv_rows(val_predictions_path, val_predictions, prediction_fields)
    write_csv_rows(test_predictions_path, test_predictions, prediction_fields)

    print(f"device: {device}")
    print(f"model: {model_path}")
    print(f"metrics: {metrics_path}")
    print(f"val predictions: {val_predictions_path}")
    print(f"test predictions: {test_predictions_path}")
    return report, val_predictions, test_predictions


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
