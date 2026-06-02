#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL_TO_INDEX = {"no_burst": 0, "burst": 1}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CPU baseline MLP on envelope_db features.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"), help="Dataset directory.")
    parser.add_argument("--model-out", type=Path, default=Path("models/envelope_mlp.joblib"), help="Output model path.")
    parser.add_argument("--report-out", type=Path, default=Path("reports/envelope_mlp_metrics.json"), help="Output metrics JSON.")
    parser.add_argument("--predictions-out", type=Path, default=Path("reports/envelope_mlp_predictions.csv"), help="Output test predictions CSV.")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64], help="Hidden layer sizes.")
    parser.add_argument("--max-iter", type=int, default=300, help="Maximum optimizer iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def read_metadata(dataset_dir: Path) -> list[dict[str, str]]:
    metadata_path = dataset_dir / "metadata.csv"
    with metadata_path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_rows(dataset_dir: Path, rows: list[dict[str, str]], split: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    selected = [row for row in rows if row.get("split") == split and row.get("label") in LABEL_TO_INDEX]
    if not selected:
        raise ValueError(f"No labeled rows found for split={split}")

    features: list[np.ndarray] = []
    labels: list[int] = []
    for row in selected:
        npz_path = dataset_dir / Path(row["npz_path"].replace("\\", "/"))
        data = np.load(npz_path)
        features.append(data["envelope_db"].astype(np.float32))
        labels.append(LABEL_TO_INDEX[row["label"]])

    lengths = {feature.shape[0] for feature in features}
    if len(lengths) != 1:
        raise ValueError(f"Envelope lengths differ: {sorted(lengths)}")

    return np.stack(features), np.asarray(labels, dtype=np.int64), selected


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, split: str) -> dict[str, object]:
    return {
        "split": split,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=[INDEX_TO_LABEL[0], INDEX_TO_LABEL[1]],
            zero_division=0,
            output_dict=True,
        ),
    }


def write_predictions(path: Path, rows: list[dict[str, str]], y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["sample_id", "source_file", "segment_start", "segment_end", "label", "predicted", "prob_no_burst", "prob_burst"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row, true_idx, pred_idx, row_proba in zip(rows, y_true, y_pred, proba):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "source_file": row["source_file"],
                    "segment_start": row["segment_start"],
                    "segment_end": row["segment_end"],
                    "label": INDEX_TO_LABEL[int(true_idx)],
                    "predicted": INDEX_TO_LABEL[int(pred_idx)],
                    "prob_no_burst": f"{row_proba[0]:.6f}",
                    "prob_burst": f"{row_proba[1]:.6f}",
                }
            )


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    rows = read_metadata(dataset_dir)

    x_train, y_train, _ = load_rows(dataset_dir, rows, "train")
    x_val, y_val, _ = load_rows(dataset_dir, rows, "val")
    x_test, y_test, test_rows = load_rows(dataset_dir, rows, "test")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=tuple(args.hidden),
                    max_iter=args.max_iter,
                    random_state=args.seed,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    test_proba = model.predict_proba(x_test)

    report = {
        "model": "envelope_mlp",
        "hidden_layer_sizes": args.hidden,
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "val": metrics_dict(y_val, val_pred, "val"),
        "test": metrics_dict(y_test, test_pred, "test"),
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_predictions(args.predictions_out, test_rows, y_test, test_pred, test_proba)

    print(f"Train samples : {len(y_train)}")
    print(f"Val samples   : {len(y_val)}")
    print(f"Test samples  : {len(y_test)}")
    print(f"Val F1        : {report['val']['f1']:.4f}")
    print(f"Test F1       : {report['test']['f1']:.4f}")
    print(f"Model         : {args.model_out}")
    print(f"Metrics       : {args.report_out}")
    print(f"Predictions   : {args.predictions_out}")


if __name__ == "__main__":
    main()
