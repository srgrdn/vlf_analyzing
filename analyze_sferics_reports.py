#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build summary statistics and plots from reports/sferics_summary.csv "
            "and reports/sferics_events.csv."
        )
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory containing sferics CSV reports. Default: ./reports",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated analytics. Default: <reports-dir>/analysis",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots in a window in addition to saving them.",
    )
    return parser.parse_args()


def configure_matplotlib(show: bool):
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def parse_float(value: str) -> float | None:
    return None if value == "" else float(value)


def parse_int(value: str) -> int:
    return int(value)


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_summary_json(output_path: Path, payload: dict[str, object]) -> None:
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_summary_txt(output_path: Path, lines: list[str]) -> None:
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_channel_totals(plt, output_dir: Path, total_ns: int, total_we: int, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    labels = ["ns", "we"]
    values = [total_ns, total_we]
    colors = ["#2563eb", "#ea580c"]
    ax.bar(labels, values, color=colors)
    ax.set_title("Total Sferics by Channel")
    ax.set_ylabel("Detections")
    for idx, value in enumerate(values):
        ax.text(idx, value + max(values) * 0.02 if max(values) else 0.1, str(value), ha="center")
    fig.savefig(output_dir / "channel_totals.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_counts_per_file(
    plt,
    output_dir: Path,
    times: list[datetime],
    count_ns: list[int],
    count_we: list[int],
    show: bool,
) -> None:
    labels = [dt.strftime("%H:%M:%S") for dt in times]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.bar(x, count_ns, label="ns", color="#2563eb")
    ax.bar(x, count_we, bottom=count_ns, label="we", color="#ea580c")
    if labels:
        max_ticks = 12
        step = max(1, int(np.ceil(len(labels) / max_ticks)))
        tick_indices = x[::step]
        if tick_indices[-1] != x[-1]:
            tick_indices = np.append(tick_indices, x[-1])
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([labels[idx] for idx in tick_indices], rotation=30, ha="right")
    ax.set_ylabel("Detections")
    ax.set_title("Detections per Minute File")
    ax.legend()
    fig.savefig(output_dir / "counts_per_file.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_histogram_per_file(plt, output_dir: Path, totals: list[int], show: bool) -> None:
    bins = np.arange(0, max(totals) + 2) - 0.5 if totals else np.arange(-0.5, 1.5, 1.0)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(totals, bins=bins, color="#16a34a", alpha=0.85, edgecolor="black")
    ax.set_xlabel("Detections per file")
    ax.set_ylabel("Files")
    ax.set_title("Histogram of Detections per Minute File")
    fig.savefig(output_dir / "histogram_per_file.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_event_timeline(
    plt,
    output_dir: Path,
    event_times: list[datetime],
    channels: list[str],
    peak_db: list[float],
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    y_map = {"ns": 0, "we": 1}
    colors = {"ns": "#2563eb", "we": "#ea580c"}
    sizes = [max(20.0, (value - min(peak_db) + 1.0) * 10.0) for value in peak_db] if peak_db else []
    for channel_name in ("ns", "we"):
        indices = [idx for idx, value in enumerate(channels) if value == channel_name]
        if not indices:
            continue
        ax.scatter(
            [event_times[idx] for idx in indices],
            [y_map[channel_name]] * len(indices),
            s=[sizes[idx] for idx in indices],
            color=colors[channel_name],
            alpha=0.8,
            label=channel_name,
        )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["ns", "we"])
    ax.set_xlabel("Absolute event time")
    ax.set_title("Timeline of Detected Sferics")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "event_timeline.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_peak_levels(plt, output_dir: Path, events: list[dict[str, str]], show: bool) -> None:
    levels_by_channel: dict[str, list[float]] = {"ns": [], "we": []}
    for row in events:
        levels_by_channel[row["channel"]].append(float(row["peak_db"]))

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    labels = []
    values = []
    for channel_name in ("ns", "we"):
        if levels_by_channel[channel_name]:
            labels.append(channel_name)
            values.append(levels_by_channel[channel_name])
    if values:
        ax.boxplot(values, labels=labels)
    ax.set_ylabel("Peak level, dB")
    ax.set_title("Peak Level Distribution by Channel")
    fig.savefig(output_dir / "peak_levels_by_channel.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    reports_dir = args.reports_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else reports_dir / "analysis"
    ensure_output_dir(output_dir)

    summary_rows = read_csv_rows(reports_dir / "sferics_summary.csv")
    event_rows = read_csv_rows(reports_dir / "sferics_events.csv")
    per_channel_rows = read_csv_rows(reports_dir / "sferics_per_channel.csv")

    total_ns = sum(parse_int(row["count_ns"]) for row in summary_rows)
    total_we = sum(parse_int(row["count_we"]) for row in summary_rows)
    total_events = sum(parse_int(row["count_total"]) for row in summary_rows)
    empty_files = sum(parse_int(row["is_empty"]) for row in summary_rows)
    active_files = len(summary_rows) - empty_files
    dominant_counter = Counter(row["dominant_channel"] for row in summary_rows)

    times = [parse_dt(row["file_time"]) for row in summary_rows]
    count_ns = [parse_int(row["count_ns"]) for row in summary_rows]
    count_we = [parse_int(row["count_we"]) for row in summary_rows]
    totals = [parse_int(row["count_total"]) for row in summary_rows]

    event_times = [parse_dt(row["peak_time_absolute"]) for row in event_rows if row["peak_time_absolute"]]
    event_channels = [row["channel"] for row in event_rows if row["peak_time_absolute"]]
    event_peak_db = [float(row["peak_db"]) for row in event_rows if row["peak_time_absolute"]]

    channel_share = {
        "ns_fraction": (total_ns / total_events) if total_events else 0.0,
        "we_fraction": (total_we / total_events) if total_events else 0.0,
    }

    summary_payload = {
        "files_total": len(summary_rows),
        "files_active": active_files,
        "files_empty": empty_files,
        "detections_total": total_events,
        "detections_ns": total_ns,
        "detections_we": total_we,
        "dominant_channel_counts": dict(dominant_counter),
        "channel_share": channel_share,
    }
    save_summary_json(output_dir / "summary_stats.json", summary_payload)

    lines = [
        f"Total files           : {len(summary_rows)}",
        f"Active files          : {active_files}",
        f"Empty files           : {empty_files}",
        f"Total detections      : {total_events}",
        f"Detections ns         : {total_ns}",
        f"Detections we         : {total_we}",
        f"Share ns              : {channel_share['ns_fraction'] * 100:.2f}%",
        f"Share we              : {channel_share['we_fraction'] * 100:.2f}%",
        f"Dominant channels     : {dict(dominant_counter)}",
        f"Per-channel rows      : {len(per_channel_rows)}",
        f"Event rows            : {len(event_rows)}",
    ]
    save_summary_txt(output_dir / "summary_stats.txt", lines)

    plt = configure_matplotlib(show=args.show)
    plot_channel_totals(plt, output_dir, total_ns, total_we, args.show)
    plot_counts_per_file(plt, output_dir, times, count_ns, count_we, args.show)
    plot_histogram_per_file(plt, output_dir, totals, args.show)
    plot_peak_levels(plt, output_dir, event_rows, args.show)
    if event_times:
        plot_event_timeline(plt, output_dir, event_times, event_channels, event_peak_db, args.show)

    print(f"Reports directory : {reports_dir}")
    print(f"Analysis output   : {output_dir}")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
