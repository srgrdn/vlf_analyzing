#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build summary statistics and plots from sferics_per_channel.csv, "
            "sferics_summary.csv, and sferics_events.csv."
        )
    )
    parser.add_argument("--reports-dir", type=Path, required=True, help="Directory with sferics_*.csv files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for plots and summary files. Default: <reports-dir>/analysis")
    return parser.parse_args()


def read_reports(reports_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_channel = pd.read_csv(reports_dir / "sferics_per_channel.csv")
    summary = pd.read_csv(reports_dir / "sferics_summary.csv")
    events = pd.read_csv(reports_dir / "sferics_events.csv")
    if "file_time" in per_channel.columns:
        per_channel["file_time"] = pd.to_datetime(per_channel["file_time"], errors="coerce")
    if "file_time" in summary.columns:
        summary["file_time"] = pd.to_datetime(summary["file_time"], errors="coerce")
    if "peak_time_absolute" in events.columns:
        events["peak_time_absolute"] = pd.to_datetime(events["peak_time_absolute"], errors="coerce")
    if "file_time" in events.columns:
        events["file_time"] = pd.to_datetime(events["file_time"], errors="coerce")
    return per_channel, summary, events


def save_summary(output_dir: Path, summary_payload: dict[str, object]) -> None:
    text_lines = [
        f"files: {summary_payload['files']}",
        f"empty_files: {summary_payload['empty_files']}",
        f"non_empty_files: {summary_payload['non_empty_files']}",
        f"total_events: {summary_payload['total_events']}",
        f"ns_total: {summary_payload['ns_total']}",
        f"we_total: {summary_payload['we_total']}",
        f"dominant_channel_counts: {summary_payload['dominant_channel_counts']}",
    ]
    (output_dir / "summary_stats.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    (output_dir / "summary_stats.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_channel_totals_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    totals = {
        "ns": float(summary["count_ns"].sum()),
        "we": float(summary["count_we"].sum()),
    }
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(list(totals.keys()), list(totals.values()), color=["#2563eb", "#ea580c"])
    ax.set_title("Total Sferics by Channel")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Detections")
    fig.savefig(output_dir / "channel_totals.png", dpi=160)
    plt.close(fig)


def build_counts_per_file_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary.sort_values("file_time").reset_index(drop=True)
    labels = ordered["file_time"].dt.strftime("%H:%M").fillna(ordered["file_name"])
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.plot(ordered.index, ordered["count_total"], marker="o", linewidth=1.5, color="#2563eb", label="total")
    ax.plot(ordered.index, ordered["count_ns"], linewidth=1.0, color="#16a34a", alpha=0.8, label="ns")
    ax.plot(ordered.index, ordered["count_we"], linewidth=1.0, color="#ea580c", alpha=0.8, label="we")
    step = max(1, len(ordered) // 12)
    tick_positions = list(range(0, len(ordered), step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([labels.iloc[idx] for idx in tick_positions], rotation=45, ha="right")
    ax.set_title("Sferics per Minute File")
    ax.set_xlabel("Minute file")
    ax.set_ylabel("Detections")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "counts_per_file.png", dpi=160)
    plt.close(fig)


def build_histogram(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(summary["count_total"], bins=min(30, max(5, len(summary) // 4)), color="#7c3aed", alpha=0.85, edgecolor="white")
    ax.set_title("Histogram of Sferics per Minute File")
    ax.set_xlabel("Detections in minute file")
    ax.set_ylabel("Files")
    ax.grid(True, alpha=0.2)
    fig.savefig(output_dir / "histogram_per_file.png", dpi=160)
    plt.close(fig)


def build_event_timeline(events: pd.DataFrame, output_dir: Path) -> None:
    if events.empty or events["peak_time_absolute"].isna().all():
        return
    ordered = events.sort_values("peak_time_absolute").copy()
    channel_offsets = ordered["channel"].map({"ns": 0, "we": 1}).fillna(-1)
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.scatter(
        ordered["peak_time_absolute"],
        channel_offsets,
        c=ordered["channel"].map({"ns": "#16a34a", "we": "#ea580c"}).fillna("#64748b"),
        s=18,
        alpha=0.8,
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["ns", "we"])
    ax.set_title("Event Timeline by Channel")
    ax.set_xlabel("Absolute time")
    ax.set_ylabel("Channel")
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.savefig(output_dir / "event_timeline.png", dpi=160)
    plt.close(fig)


def build_peak_levels_plot(events: pd.DataFrame, output_dir: Path) -> None:
    if events.empty or "peak_db" not in events.columns:
        return
    cleaned = events.copy()
    cleaned["peak_db"] = pd.to_numeric(cleaned["peak_db"], errors="coerce")
    cleaned = cleaned.dropna(subset=["peak_db"])
    if cleaned.empty:
        return
    ns = cleaned.loc[cleaned["channel"] == "ns", "peak_db"]
    we = cleaned.loc[cleaned["channel"] == "we", "peak_db"]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    data = [series.tolist() for series in [ns, we] if not series.empty]
    labels = [label for label, series in [("ns", ns), ("we", we)] if not series.empty]
    ax.boxplot(data, tick_labels=labels)
    ax.set_title("Peak Level Distribution by Channel")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Peak value, dB")
    ax.grid(True, alpha=0.2)
    fig.savefig(output_dir / "peak_levels_by_channel.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    reports_dir = args.reports_dir.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else reports_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    per_channel, summary, events = read_reports(reports_dir)
    summary_payload = {
        "files": int(len(summary)),
        "empty_files": int(summary["is_empty"].sum()) if "is_empty" in summary.columns else 0,
        "non_empty_files": int((summary["is_empty"] == 0).sum()) if "is_empty" in summary.columns else int(len(summary)),
        "total_events": int(len(events)),
        "ns_total": int(summary["count_ns"].sum()) if "count_ns" in summary.columns else 0,
        "we_total": int(summary["count_we"].sum()) if "count_we" in summary.columns else 0,
        "dominant_channel_counts": summary["dominant_channel"].value_counts(dropna=False).to_dict() if "dominant_channel" in summary.columns else {},
    }
    save_summary(output_dir, summary_payload)
    build_channel_totals_plot(summary, output_dir)
    build_counts_per_file_plot(summary, output_dir)
    build_histogram(summary, output_dir)
    build_event_timeline(events, output_dir)
    build_peak_levels_plot(events, output_dir)

    print(f"Reports dir      : {reports_dir}")
    print(f"Analysis dir     : {output_dir}")
    print(f"Files            : {summary_payload['files']}")
    print(f"Empty files      : {summary_payload['empty_files']}")
    print(f"Non-empty files  : {summary_payload['non_empty_files']}")
    print(f"Total events     : {summary_payload['total_events']}")


if __name__ == "__main__":
    main()
