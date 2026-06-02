#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from envelope_model_inference import (
    DEFAULT_BURST_THRESHOLD,
    DEFAULT_CHANNELS,
    DEFAULT_FILE_PATTERN,
    DEFAULT_FREQUENCY_RESOLUTION,
    DEFAULT_FREQ_MAX,
    DEFAULT_FREQ_MIN,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SEGMENT_DURATION,
    DEFAULT_TIME_RESOLUTION,
    analyze_file_with_envelope_model,
    collect_input_files,
    load_envelope_model,
    segment_rows_to_dicts,
    summary_to_dict,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an envelope_1d PyTorch model on raw Broadband_Data *.bin files. "
            "Each minute file is split into 2-second windows, transformed to envelope_db, "
            "and classified as burst/no_burst."
        )
    )
    parser.add_argument("input", type=Path, help="Path to Broadband_Data_*.bin or a directory with such files.")
    parser.add_argument("--model", type=Path, required=True, help="Path to a .pt checkpoint such as models/envelope_1d_dataset_v2.pt.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/model_envelope_1d"), help="Directory for CSV reports.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subdirectories.")
    parser.add_argument("--file-pattern", default=DEFAULT_FILE_PATTERN, help=f"Input filename glob. Default: {DEFAULT_FILE_PATTERN}")
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--channel", choices=("ns", "we", "both"), default="ns", help="Model channel to analyze. Use both to run ns and we. Default: ns")
    parser.add_argument("--segment-duration", type=float, default=DEFAULT_SEGMENT_DURATION)
    parser.add_argument("--time-resolution", type=float, default=DEFAULT_TIME_RESOLUTION)
    parser.add_argument("--frequency-resolution", type=float, default=DEFAULT_FREQUENCY_RESOLUTION)
    parser.add_argument("--freq-min", type=float, default=DEFAULT_FREQ_MIN)
    parser.add_argument("--freq-max", type=float, default=DEFAULT_FREQ_MAX)
    parser.add_argument("--burst-threshold", type=float, default=DEFAULT_BURST_THRESHOLD, help="Probability threshold for burst classification. Default: 0.5")
    parser.add_argument("--max-files", type=int, default=None, help="Limit the number of processed files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    files = collect_input_files(input_path, recursive=args.recursive, file_pattern=args.file_pattern)
    if args.max_files is not None:
        files = files[: args.max_files]

    model_kind, model = load_envelope_model(args.model)
    segment_rows: list[dict[str, object]] = []
    per_channel_summary_rows: list[dict[str, object]] = []
    combined_summary_rows: list[dict[str, object]] = []
    channels = ("ns", "we") if args.channel == "both" else (args.channel,)

    for index, file_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {file_path.name}")
        channel_summaries = {}
        for channel in channels:
            per_segment, summary = analyze_file_with_envelope_model(
                file_path=file_path,
                model=model,
                channel=channel,
                sample_rate=args.sample_rate,
                channels=args.channels,
                segment_duration=args.segment_duration,
                time_resolution=args.time_resolution,
                frequency_resolution=args.frequency_resolution,
                freq_min=args.freq_min,
                freq_max=args.freq_max,
                burst_threshold=args.burst_threshold,
            )
            segment_rows.extend(segment_rows_to_dicts(per_segment))
            per_channel_summary_rows.append(summary_to_dict(summary))
            channel_summaries[channel] = summary

        if args.channel == "both":
            ns = channel_summaries["ns"]
            we = channel_summaries["we"]
            combined_summary_rows.append(
                {
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_time": ns.file_time or we.file_time,
                    "duration_s": f"{max(ns.duration_s, we.duration_s):.6f}",
                    "segment_count": max(ns.segment_count, we.segment_count),
                    "burst_segment_count_ns": ns.burst_segment_count,
                    "burst_segment_count_we": we.burst_segment_count,
                    "burst_segment_count_total": ns.burst_segment_count + we.burst_segment_count,
                    "max_prob_burst_ns": f"{ns.max_prob_burst:.6f}",
                    "max_prob_burst_we": f"{we.max_prob_burst:.6f}",
                    "mean_prob_burst_ns": f"{ns.mean_prob_burst:.6f}",
                    "mean_prob_burst_we": f"{we.mean_prob_burst:.6f}",
                    "dominant_channel": (
                        "ns"
                        if ns.burst_segment_count > we.burst_segment_count
                        else "we"
                        if we.burst_segment_count > ns.burst_segment_count
                        else "equal"
                    ),
                    "route_decision": "burst_detected" if (ns.burst_segment_count or we.burst_segment_count) else "empty",
                }
            )

    segment_csv = output_dir / "segments.csv"
    summary_csv = output_dir / "summary.csv"
    write_csv(segment_csv, segment_rows)
    write_csv(summary_csv, per_channel_summary_rows)
    if combined_summary_rows:
        write_csv(output_dir / "summary_combined.csv", combined_summary_rows)

    total_segments = sum(int(row["segment_count"]) for row in per_channel_summary_rows) if per_channel_summary_rows else 0
    total_burst_segments = sum(int(row["burst_segment_count"]) for row in per_channel_summary_rows) if per_channel_summary_rows else 0

    print(f"Model kind      : {model_kind}")
    print(f"Input files     : {len(files)}")
    print(f"Channels        : {', '.join(channels)}")
    print(f"Total segments  : {total_segments}")
    print(f"Burst segments  : {total_burst_segments}")
    print(f"Segments CSV    : {segment_csv}")
    print(f"Summary CSV     : {summary_csv}")
    if combined_summary_rows:
        print(f"Combined CSV    : {output_dir / 'summary_combined.csv'}")


if __name__ == "__main__":
    main()
