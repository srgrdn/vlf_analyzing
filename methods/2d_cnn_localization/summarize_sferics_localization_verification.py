#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METHOD_DIR = Path(__file__).resolve().parent
DEFAULT_VERIFICATION_CSV = METHOD_DIR / "reports" / "raw_verification" / "sferics_localization_2d_raw_verification.csv"
DEFAULT_OUTPUT_DIR = METHOD_DIR / "reports" / "raw_verification"
METRIC_STATUSES = {"model_verified", "manual_corrected", "false_positive", "missed_events"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize interactive raw-file verification for the 2D sferics "
            "localization model. Computes event-level metrics and optional plots."
        )
    )
    parser.add_argument(
        "--verification-csv",
        type=Path,
        default=DEFAULT_VERIFICATION_CSV,
        help=f"Verification CSV. Default: {DEFAULT_VERIFICATION_CSV}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output dir for JSON/plots. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=0.04,
        help="Event matching tolerance in seconds. Default: 0.04",
    )
    parser.add_argument("--save-plots", action="store_true", help="Save summary PNG plots.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    return parser.parse_args()


def configure_matplotlib(show: bool):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Verification CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def parse_times(raw: str | None) -> list[float]:
    text = (raw or "").strip()
    if not text:
        return []
    values = json.loads(text)
    if not isinstance(values, list):
        raise ValueError("Expected JSON list of event times.")
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


def empty_metrics() -> dict[str, Any]:
    return {
        "segments": 0,
        "event_tp": 0,
        "event_fp": 0,
        "event_fn": 0,
        "event_precision": 0.0,
        "event_recall": 0.0,
        "event_f1": 0.0,
        "event_count_mae": 0.0,
        "mean_predicted_count": 0.0,
        "mean_corrected_count": 0.0,
    }


def finalize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return empty_metrics()
    tp = sum(int(row["event_tp"]) for row in rows)
    fp = sum(int(row["event_fp"]) for row in rows)
    fn = sum(int(row["event_fn"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pred_counts = np.asarray([int(row["predicted_count"]) for row in rows], dtype=np.float32)
    true_counts = np.asarray([int(row["corrected_count"]) for row in rows], dtype=np.float32)
    return {
        "segments": len(rows),
        "event_tp": int(tp),
        "event_fp": int(fp),
        "event_fn": int(fn),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "event_count_mae": float(np.mean(np.abs(pred_counts - true_counts))),
        "mean_predicted_count": float(np.mean(pred_counts)),
        "mean_corrected_count": float(np.mean(true_counts)),
    }


def compute_summary(rows: list[dict[str, str]], match_tolerance: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reviewed_rows = [row for row in rows if (row.get("review_status") or "").strip()]
    metric_rows = [
        row
        for row in reviewed_rows
        if (row.get("review_status") or "").strip() in METRIC_STATUSES
        and not (row.get("quality_flag") or "").strip()
    ]

    per_segment: list[dict[str, Any]] = []
    for row in metric_rows:
        predicted = parse_times(row.get("predicted_event_times_inside_s_json"))
        corrected = parse_times(row.get("corrected_event_times_inside_s_json"))
        tp, fp, fn = match_event_times(corrected, predicted, match_tolerance)
        per_segment.append(
            {
                "verification_id": row["verification_id"],
                "source_file": row["source_file"],
                "channel": row["channel"],
                "review_status": row["review_status"],
                "predicted_count": len(predicted),
                "corrected_count": len(corrected),
                "count_error": len(predicted) - len(corrected),
                "event_tp": tp,
                "event_fp": fp,
                "event_fn": fn,
                "max_heatmap": float(row.get("max_heatmap") or 0.0),
            }
        )

    by_channel_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_status_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_segment:
        by_channel_rows[str(row["channel"])].append(row)
        by_status_rows[str(row["review_status"])].append(row)

    status_counts = Counter((row.get("review_status") or "pending").strip() or "pending" for row in rows)
    channel_counts = Counter(row.get("channel", "") for row in rows)
    summary = {
        "verification_rows": len(rows),
        "reviewed_rows": len(reviewed_rows),
        "pending_rows": len(rows) - len(reviewed_rows),
        "metric_rows": len(metric_rows),
        "match_tolerance": match_tolerance,
        "status_counts": dict(sorted(status_counts.items())),
        "channel_counts": dict(sorted(channel_counts.items())),
        "overall": finalize_metrics(per_segment),
        "by_channel": {
            channel: finalize_metrics(channel_rows)
            for channel, channel_rows in sorted(by_channel_rows.items())
        },
        "by_status": {
            status: finalize_metrics(status_rows)
            for status, status_rows in sorted(by_status_rows.items())
        },
    }
    return summary, per_segment


def write_segment_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(plt, output_dir: Path, summary: dict[str, Any], per_segment: list[dict[str, Any]], show: bool) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    status_counts = summary["status_counts"]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(list(status_counts.keys()), list(status_counts.values()))
    ax.set_title("Verification status counts")
    ax.set_ylabel("Segments")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(plots_dir / "status_counts.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    if not per_segment:
        return

    channels = sorted(set(row["channel"] for row in per_segment))
    precision = [summary["by_channel"][channel]["event_precision"] for channel in channels]
    recall = [summary["by_channel"][channel]["event_recall"] for channel in channels]
    f1 = [summary["by_channel"][channel]["event_f1"] for channel in channels]
    x = np.arange(len(channels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(x - width, precision, width, label="precision")
    ax.bar(x, recall, width, label="recall")
    ax.bar(x + width, f1, width, label="f1")
    ax.set_xticks(x, channels)
    ax.set_ylim(0, 1.0)
    ax.set_title("Event metrics by channel")
    ax.legend()
    fig.savefig(plots_dir / "event_metrics_by_channel.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    count_errors = [int(row["count_error"]) for row in per_segment]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    bins = range(min(count_errors) - 1, max(count_errors) + 2)
    ax.hist(count_errors, bins=bins, edgecolor="black")
    ax.set_title("Predicted count minus corrected count")
    ax.set_xlabel("Count error")
    ax.set_ylabel("Segments")
    fig.savefig(plots_dir / "event_count_error_histogram.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    predicted = [int(row["predicted_count"]) for row in per_segment]
    corrected = [int(row["corrected_count"]) for row in per_segment]
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(corrected, predicted, alpha=0.7)
    max_count = max(max(predicted), max(corrected), 1)
    ax.plot([0, max_count], [0, max_count], color="red", linestyle="--")
    ax.set_title("Predicted vs corrected event count")
    ax.set_xlabel("Corrected count")
    ax.set_ylabel("Predicted count")
    fig.savefig(plots_dir / "predicted_vs_corrected_count.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    verification_csv = args.verification_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rows = read_rows(verification_csv)
    summary, per_segment = compute_summary(rows, args.match_tolerance)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "sferics_localization_2d_verification_summary.json"
    metrics_path = output_dir / "sferics_localization_2d_verification_segment_metrics.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_segment_metrics(metrics_path, per_segment)

    if args.save_plots or args.show:
        plt = configure_matplotlib(show=args.show)
        save_plots(plt, output_dir, summary, per_segment, args.show)

    print(f"Verification CSV : {verification_csv}")
    print(f"Reviewed rows    : {summary['reviewed_rows']}/{summary['verification_rows']}")
    print(f"Metric rows      : {summary['metric_rows']}")
    print(f"Summary          : {summary_path}")
    print(f"Segment metrics  : {metrics_path}")
    print("Overall:")
    overall = summary["overall"]
    print(
        f"  precision={overall['event_precision']:.3f} "
        f"recall={overall['event_recall']:.3f} "
        f"f1={overall['event_f1']:.3f} "
        f"count_mae={overall['event_count_mae']:.3f}"
    )
    print("By channel:")
    for channel, metrics in summary["by_channel"].items():
        print(
            f"  {channel}: segments={metrics['segments']} "
            f"precision={metrics['event_precision']:.3f} "
            f"recall={metrics['event_recall']:.3f} "
            f"f1={metrics['event_f1']:.3f} "
            f"count_mae={metrics['event_count_mae']:.3f}"
        )


if __name__ == "__main__":
    main()
