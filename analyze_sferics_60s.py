from __future__ import annotations

import argparse
import csv
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from detect_broadband_bursts import (
    aggregate_band,
    compute_spectrogram_power,
    pick_peaks,
    robust_threshold,
    smooth_series,
)
from plot_broadband_spectrogram import (
    CHANNEL_NAME_TO_INDEX,
    crop_frequency_band,
    extract_channel,
    load_samples,
    read_metadata,
    resolve_frequency_band,
    resolve_stft_parameters,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Count sferics in full 60-second broadband files.")
    parser.add_argument("input", type=Path, help="Path to .bin file or directory with .bin files.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--sample-rate", type=float, default=100000.0)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--freq-min", type=float, default=20000.0)
    parser.add_argument("--freq-max", type=float, default=30000.0)
    parser.add_argument("--time-resolution", type=float, default=0.002)
    parser.add_argument("--frequency-resolution", type=float, default=50.0)
    parser.add_argument("--threshold-mad", type=float, default=8.0)
    parser.add_argument("--min-peak-distance", type=float, default=0.01)
    parser.add_argument("--smooth-bins", type=int, default=3)
    parser.add_argument("--aggregate", choices=("mean", "sum", "max"), default="mean")
    parser.add_argument("--output", type=Path, default=Path("reports/sferics_60s_by_channel.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("reports/sferics_60s_summary.csv"))
    parser.add_argument("--match-tolerance", type=float, default=0.02, help="Peak matching tolerance between NS and WE, seconds.")
    return parser.parse_args()


def collect_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    iterator = path.rglob("*.bin") if recursive else path.glob("*.bin")
    return sorted(p for p in iterator if p.is_file())


def parse_time_from_name(path: Path) -> str:
    match = re.search(r"(\d{4})\.(\d{2})\.(\d{2})_(\d{2})\.(\d{2})\.(\d{2})", path.name)
    if not match:
        return ""
    dt = datetime(
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
        int(match.group(5)),
        int(match.group(6)),
    )
    return dt.isoformat(sep=" ")


def analyze_channel(file_path: Path, channel_name: str, args):
    record_size, sample_count, data_offset = read_metadata(file_path)
    samples = load_samples(file_path, sample_count, record_size, data_offset)

    channel_index = CHANNEL_NAME_TO_INDEX[channel_name]
    channel_samples = extract_channel(
        samples=samples,
        sample_count=sample_count,
        record_size=record_size,
        channels=args.channels,
        channel_index=channel_index,
    )

    nperseg, hop = resolve_stft_parameters(
        samples=channel_samples,
        sample_rate=args.sample_rate,
        nperseg=4096,
        max_frames=1000000,
        time_resolution=args.time_resolution,
        frequency_resolution=args.frequency_resolution,
    )

    freq_min, freq_max = resolve_frequency_band(args.sample_rate, args.freq_min, args.freq_max)

    power, time_axis, freq_axis = compute_spectrogram_power(channel_samples, nperseg=nperseg, hop=hop)
    power, freq_axis = crop_frequency_band(
        spec_db=power,
        freq_axis=freq_axis,
        sample_rate=args.sample_rate,
        freq_min=freq_min,
        freq_max=freq_max,
    )

    band_power = aggregate_band(power, args.aggregate)
    envelope_db = 10.0 * np.log10(np.maximum(band_power, 1e-12))
    envelope_db = smooth_series(envelope_db, args.smooth_bins)

    _, _, threshold = robust_threshold(envelope_db, args.threshold_mad)
    min_distance_bins = max(1, int(round(args.min_peak_distance * args.sample_rate / hop)))
    peak_indices = pick_peaks(envelope_db, threshold, min_distance_bins)

    time_s = time_axis / args.sample_rate
    duration_s = len(channel_samples) / args.sample_rate

    peak_times = time_s[peak_indices] if len(peak_indices) else np.array([], dtype=np.float32)
    peak_db = envelope_db[peak_indices] if len(peak_indices) else np.array([], dtype=np.float32)

    return {
        "channel": channel_name,
        "duration_s": duration_s,
        "count": int(len(peak_indices)),
        "rate_per_min": float(len(peak_indices) / duration_s * 60.0) if duration_s else 0.0,
        "threshold_db": float(threshold),
        "mean_peak_db": float(np.mean(peak_db)) if len(peak_db) else np.nan,
        "max_peak_db": float(np.max(peak_db)) if len(peak_db) else np.nan,
        "peak_times": peak_times,
        "peak_db": peak_db,
    }


def match_channels(ns, we, tolerance_s: float):
    ns_times = ns["peak_times"]
    we_times = we["peak_times"]
    ns_db = ns["peak_db"]
    we_db = we["peak_db"]

    matches = []
    used_we = set()

    for i, t_ns in enumerate(ns_times):
        if len(we_times) == 0:
            continue

        distances = np.abs(we_times - t_ns)
        j = int(np.argmin(distances))
        if distances[j] <= tolerance_s and j not in used_we:
            used_we.add(j)

            # Envelope power is non-negative; this gives only a rough uncalibrated direction proxy.
            amp_ns = math.sqrt(10.0 ** (float(ns_db[i]) / 10.0))
            amp_we = math.sqrt(10.0 ** (float(we_db[j]) / 10.0))
            azimuth_proxy = math.degrees(math.atan2(amp_we, amp_ns))

            matches.append(
                {
                    "time_s": float((t_ns + we_times[j]) / 2.0),
                    "ns_db": float(ns_db[i]),
                    "we_db": float(we_db[j]),
                    "we_ns_ratio_db": float(we_db[j] - ns_db[i]),
                    "azimuth_proxy_deg": float(azimuth_proxy),
                }
            )

    return matches


def main():
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    files = collect_files(input_path, args.recursive)

    if not files:
        raise ValueError(f"No .bin files found: {input_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    channel_rows = []
    summary_rows = []

    for idx, file_path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] {file_path.name}")

        file_time = parse_time_from_name(file_path)
        ns = analyze_channel(file_path, "ns", args)
        we = analyze_channel(file_path, "we", args)
        matches = match_channels(ns, we, args.match_tolerance)

        for result in [ns, we]:
            channel_rows.append(
                {
                    "file": file_path.name,
                    "file_time": file_time,
                    "channel": result["channel"],
                    "duration_s": f"{result['duration_s']:.3f}",
                    "count": result["count"],
                    "rate_per_min": f"{result['rate_per_min']:.3f}",
                    "threshold_db": f"{result['threshold_db']:.3f}",
                    "mean_peak_db": "" if np.isnan(result["mean_peak_db"]) else f"{result['mean_peak_db']:.3f}",
                    "max_peak_db": "" if np.isnan(result["max_peak_db"]) else f"{result['max_peak_db']:.3f}",
                }
            )

        dominant_channel = "ns" if ns["count"] > we["count"] else "we" if we["count"] > ns["count"] else "equal"
        summary_rows.append(
            {
                "file": file_path.name,
                "file_time": file_time,
                "duration_s": f"{ns['duration_s']:.3f}",
                "ns_count": ns["count"],
                "we_count": we["count"],
                "total_count_ns_plus_we": ns["count"] + we["count"],
                "matched_ns_we_count": len(matches),
                "dominant_channel": dominant_channel,
                "ns_rate_per_min": f"{ns['rate_per_min']:.3f}",
                "we_rate_per_min": f"{we['rate_per_min']:.3f}",
                "mean_azimuth_proxy_deg": "" if not matches else f"{np.mean([m['azimuth_proxy_deg'] for m in matches]):.3f}",
                "mean_we_ns_ratio_db": "" if not matches else f"{np.mean([m['we_ns_ratio_db'] for m in matches]):.3f}",
            }
        )

    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(channel_rows[0].keys()))
        writer.writeheader()
        writer.writerows(channel_rows)

    with args.summary_output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Channel CSV : {args.output}")
    print(f"Summary CSV : {args.summary_output}")
    print(f"Files       : {len(files)}")


if __name__ == "__main__":
    main()
