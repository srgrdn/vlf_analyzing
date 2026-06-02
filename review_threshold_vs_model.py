#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from build_burst_dataset import render_review_png
from detect_broadband_bursts import (
    aggregate_band,
    compute_spectrogram_power,
    pick_peaks,
    robust_threshold,
    smooth_series,
)
from envelope_model_inference import (
    DEFAULT_BURST_THRESHOLD,
    DEFAULT_CHANNELS,
    DEFAULT_FREQUENCY_RESOLUTION,
    DEFAULT_FREQ_MAX,
    DEFAULT_FREQ_MIN,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SEGMENT_DURATION,
    DEFAULT_TIME_RESOLUTION,
    load_envelope_model,
    predict_envelope,
)
from plot_broadband_spectrogram import (
    CHANNEL_NAME_TO_INDEX,
    configure_matplotlib,
    crop_frequency_band,
    extract_channel,
    iter_time_segments,
    load_samples,
    read_metadata,
    resolve_frequency_band,
    resolve_stft_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a manual review directory for one minute .bin file: "
            "2-second spectrogram PNGs plus a combined CSV with threshold-detector and model decisions."
        )
    )
    parser.add_argument("input", type=Path, help="Path to one Broadband_Data_*.bin file.")
    parser.add_argument("--model", type=Path, required=True, help="Path to envelope_1d .pt checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for review PNGs and CSV reports.")
    parser.add_argument("--channel", choices=("ns", "we", "both"), default="both", help="Review one or both channels. Default: both")
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--segment-duration", type=float, default=DEFAULT_SEGMENT_DURATION)
    parser.add_argument("--time-resolution", type=float, default=DEFAULT_TIME_RESOLUTION)
    parser.add_argument("--frequency-resolution", type=float, default=DEFAULT_FREQUENCY_RESOLUTION)
    parser.add_argument("--freq-min", type=float, default=DEFAULT_FREQ_MIN)
    parser.add_argument("--freq-max", type=float, default=DEFAULT_FREQ_MAX)
    parser.add_argument("--aggregate", choices=("mean", "sum", "max"), default="mean")
    parser.add_argument("--threshold-mad", type=float, default=8.0)
    parser.add_argument("--min-peak-distance", type=float, default=0.01)
    parser.add_argument("--smooth-bins", type=int, default=3)
    parser.add_argument("--burst-threshold", type=float, default=DEFAULT_BURST_THRESHOLD)
    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--db-low", type=float, default=2.0)
    parser.add_argument("--db-high", type=float, default=99.8)
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def analyze_channel(
    file_path: Path,
    model,
    channel: str,
    args: argparse.Namespace,
    plt,
    output_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    record_size, sample_count, data_offset = read_metadata(file_path)
    samples = load_samples(file_path, sample_count, record_size, data_offset)
    channel_samples = extract_channel(
        samples=samples,
        sample_count=sample_count,
        record_size=record_size,
        channels=args.channels,
        channel_index=CHANNEL_NAME_TO_INDEX[channel],
    )
    nperseg, hop = resolve_stft_parameters(
        samples=channel_samples,
        sample_rate=args.sample_rate,
        nperseg=4096,
        max_frames=4000,
        time_resolution=args.time_resolution,
        frequency_resolution=args.frequency_resolution,
    )
    freq_min, freq_max = resolve_frequency_band(args.sample_rate, args.freq_min, args.freq_max)
    min_distance_bins = max(1, int(round(args.min_peak_distance * args.sample_rate / hop)))
    segments = iter_time_segments(
        samples=channel_samples,
        sample_rate=args.sample_rate,
        segment_duration=args.segment_duration,
        nperseg=nperseg,
    )

    rows: list[dict[str, object]] = []
    burst_segments_model = 0
    burst_segments_threshold = 0
    probs: list[float] = []
    threshold_counts: list[int] = []

    channel_dir = output_dir / channel
    channel_dir.mkdir(parents=True, exist_ok=True)

    for segment_index, start, stop, start_s, stop_s in segments:
        segment_samples = channel_samples[start:stop]
        power, time_axis, freq_axis = compute_spectrogram_power(segment_samples, nperseg=nperseg, hop=hop)
        power, freq_axis = crop_frequency_band(
            spec_db=power,
            freq_axis=freq_axis,
            sample_rate=args.sample_rate,
            freq_min=freq_min,
            freq_max=freq_max,
        )
        spec_db = 10.0 * np.log10(np.maximum(power, 1e-12)).astype(np.float32)
        envelope_power = aggregate_band(power, args.aggregate)
        envelope_db = 10.0 * np.log10(np.maximum(envelope_power, 1e-12))
        envelope_db = smooth_series(envelope_db, args.smooth_bins).astype(np.float32)
        _, _, threshold = robust_threshold(envelope_db, args.threshold_mad)
        peak_indices = pick_peaks(envelope_db, threshold, min_distance_bins)
        threshold_detected = bool(len(peak_indices))
        threshold_peak_count = int(len(peak_indices))
        model_label, prob_no_burst, prob_burst = predict_envelope(model, envelope_db)
        if prob_burst >= args.burst_threshold:
            model_label = "burst"
            burst_segments_model += 1
        else:
            model_label = "no_burst"
        if threshold_detected:
            burst_segments_threshold += 1

        probs.append(prob_burst)
        threshold_counts.append(threshold_peak_count)

        time_axis_s = (time_axis / args.sample_rate).astype(np.float32)
        freq_axis_hz = (freq_axis * args.sample_rate).astype(np.float32)
        png_path = channel_dir / f"{file_path.stem}_{channel}_seg{segment_index:03d}_{start_s:07.3f}-{stop_s:07.3f}s.png"
        title = (
            f"{file_path.name} | {channel} | seg {segment_index:03d} | {start_s:.3f}-{stop_s:.3f}s | "
            f"threshold={threshold_peak_count} peaks | model={model_label} ({prob_burst:.3f})"
        )
        render_review_png(
            plt=plt,
            spec_db=spec_db,
            time_axis_s=time_axis_s,
            freq_axis_hz=freq_axis_hz,
            output_path=png_path,
            title=title,
            cmap=args.cmap,
            db_low=args.db_low,
            db_high=args.db_high,
            dpi=args.dpi,
        )

        rows.append(
            {
                "file_name": file_path.name,
                "channel": channel,
                "segment_index": segment_index,
                "segment_start_s": f"{start_s:.6f}",
                "segment_end_s": f"{stop_s:.6f}",
                "threshold_peak_count": threshold_peak_count,
                "threshold_detected": int(threshold_detected),
                "threshold_db": f"{threshold:.6f}",
                "model_label": model_label,
                "prob_no_burst": f"{prob_no_burst:.6f}",
                "prob_burst": f"{prob_burst:.6f}",
                "peak_value_db": f"{float(np.max(envelope_db)):.6f}",
                "mean_value_db": f"{float(np.mean(envelope_db)):.6f}",
                "png_path": str(png_path),
            }
        )

    summary = {
        "file_name": file_path.name,
        "channel": channel,
        "segment_count": len(segments),
        "threshold_burst_segments": burst_segments_threshold,
        "model_burst_segments": burst_segments_model,
        "mean_prob_burst": f"{float(np.mean(probs)) if probs else 0.0:.6f}",
        "max_prob_burst": f"{max(probs) if probs else 0.0:.6f}",
        "mean_threshold_peak_count": f"{float(np.mean(threshold_counts)) if threshold_counts else 0.0:.6f}",
    }
    return rows, summary


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
    file_path = args.input.expanduser().resolve()
    if not file_path.is_file():
        raise ValueError(f"Input file does not exist: {file_path}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_kind, model = load_envelope_model(args.model)
    plt = configure_matplotlib(show=False)
    channels = ("ns", "we") if args.channel == "both" else (args.channel,)

    combined_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for channel in channels:
        rows, summary = analyze_channel(file_path, model, channel, args, plt, output_dir)
        combined_rows.extend(rows)
        summary_rows.append(summary)

    write_csv(output_dir / "segments_review.csv", combined_rows)
    write_csv(output_dir / "summary_per_channel.csv", summary_rows)
    if args.channel == "both" and len(summary_rows) == 2:
        ns, we = summary_rows
        write_csv(
            output_dir / "summary_combined.csv",
            [
                {
                    "file_name": file_path.name,
                    "segment_count": ns["segment_count"],
                    "threshold_burst_segments_ns": ns["threshold_burst_segments"],
                    "threshold_burst_segments_we": we["threshold_burst_segments"],
                    "model_burst_segments_ns": ns["model_burst_segments"],
                    "model_burst_segments_we": we["model_burst_segments"],
                    "dominant_channel_model": (
                        "ns"
                        if int(ns["model_burst_segments"]) > int(we["model_burst_segments"])
                        else "we"
                        if int(we["model_burst_segments"]) > int(ns["model_burst_segments"])
                        else "equal"
                    ),
                    "dominant_channel_threshold": (
                        "ns"
                        if int(ns["threshold_burst_segments"]) > int(we["threshold_burst_segments"])
                        else "we"
                        if int(we["threshold_burst_segments"]) > int(ns["threshold_burst_segments"])
                        else "equal"
                    ),
                }
            ],
        )

    print(f"Model kind       : {model_kind}")
    print(f"Input file       : {file_path}")
    print(f"Channels         : {', '.join(channels)}")
    print(f"Review output    : {output_dir}")
    print(f"Segments CSV     : {output_dir / 'segments_review.csv'}")
    print(f"Summary CSV      : {output_dir / 'summary_per_channel.csv'}")
    if args.channel == 'both':
        print(f"Combined CSV     : {output_dir / 'summary_combined.csv'}")


if __name__ == "__main__":
    main()
