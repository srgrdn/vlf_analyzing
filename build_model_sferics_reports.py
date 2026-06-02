#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert run_envelope_1d_model outputs into threshold-style report tables: "
            "sferics_per_channel.csv, sferics_summary.csv, and sferics_events.csv."
        )
    )
    parser.add_argument("--model-report-dir", type=Path, required=True, help="Directory produced by run_envelope_1d_model.py.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for converted report CSV files.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    report_dir = args.model_report_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    segments = read_csv(report_dir / "segments.csv")
    per_channel = read_csv(report_dir / "summary.csv")
    combined = read_csv(report_dir / "summary_combined.csv")

    per_channel_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []

    for row in per_channel:
        burst_segments = int(row["burst_segment_count"])
        per_channel_rows.append(
            {
                "file_name": row["file_name"],
                "file_time": row["file_time"],
                "channel": row["channel"],
                "detections_count": burst_segments,
                "max_peak_db": row["max_segment_peak_db"] if burst_segments else "",
                "mean_peak_db": "",
                "median_peak_db": "",
                "status": row["route_decision"],
                "final_path": row.get("final_path", ""),
            }
        )

    for row in combined:
        count_ns = int(row["burst_segment_count_ns"])
        count_we = int(row["burst_segment_count_we"])
        total = int(row["burst_segment_count_total"])
        summary_rows.append(
            {
                "file_name": row["file_name"],
                "file_time": row["file_time"],
                "count_ns": count_ns,
                "count_we": count_we,
                "count_total": total,
                "max_level_ns": "",
                "max_level_we": "",
                "dominant_channel": row["dominant_channel"],
                "is_empty": 1 if total == 0 else 0,
                "final_route": row["route_decision"],
                "processing1_path": "",
                "processing2_path": "",
            }
        )

    for row in segments:
        if row["predicted_label"] != "burst":
            continue
        start_s = float(row["segment_start"])
        end_s = float(row["segment_end"])
        midpoint_s = (start_s + end_s) / 2.0
        file_time = row["file_time"]
        absolute = ""
        if file_time:
            try:
                absolute_dt = datetime.fromisoformat(file_time) + timedelta(seconds=midpoint_s)
                absolute = absolute_dt.isoformat(sep=" ")
            except ValueError:
                absolute = ""
        event_rows.append(
            {
                "file_name": row["file_name"],
                "file_time": file_time,
                "channel": row["channel"],
                "peak_time_local_s": f"{midpoint_s:.6f}",
                "peak_time_absolute": absolute,
                "peak_time_global_s": f"{midpoint_s:.6f}",
                "peak_db": row["peak_value_db"],
                "threshold_db": "",
                "model_prob_burst": row["prob_burst"],
                "segment_index": row["segment_index"],
                "segment_start_s": row["segment_start"],
                "segment_stop_s": row["segment_end"],
            }
        )

    write_csv(output_dir / "sferics_per_channel.csv", per_channel_rows)
    write_csv(output_dir / "sferics_summary.csv", summary_rows)
    write_csv(output_dir / "sferics_events.csv", event_rows)

    print(f"Input report dir : {report_dir}")
    print(f"Output dir       : {output_dir}")
    print(f"Per-channel rows : {len(per_channel_rows)}")
    print(f"Summary rows     : {len(summary_rows)}")
    print(f"Event rows       : {len(event_rows)}")


if __name__ == "__main__":
    main()
