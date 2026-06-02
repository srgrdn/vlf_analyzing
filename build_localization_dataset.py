#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
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
    configure_matplotlib,
    crop_frequency_band,
    extract_channel,
    iter_time_segments,
    load_samples,
    read_metadata,
    resolve_frequency_band,
    resolve_stft_parameters,
)


DEFAULT_PATTERN = "Broadband_Data_*.bin"
DEFAULT_LABEL_SOURCE = "threshold_auto"
DEFAULT_QUALITY_FLAG = "unchecked"
METADATA_FIELDNAMES = [
    "sample_id",
    "source_file",
    "channel",
    "segment_index",
    "segment_start",
    "segment_end",
    "sample_rate",
    "time_resolution",
    "frequency_resolution",
    "freq_min",
    "freq_max",
    "npz_path",
    "png_path",
    "event_count",
    "event_times_s_json",
    "split",
    "label_source",
    "quality_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a localization dataset from Broadband_Data *.bin files using the "
            "threshold detector as an auto-label source for event times."
        )
    )
    parser.add_argument("input", type=Path, help="Path to a Broadband_Data_*.bin file or directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_localization"),
        help="Dataset output directory. Default: dataset_localization",
    )
    parser.add_argument(
        "--channel",
        choices=("ns", "we", "both"),
        default="ns",
        help="Channel selector. Use both to build separate samples for ns and we.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directories.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild the output directory from scratch.")
    parser.add_argument("--sample-rate", type=float, default=100000.0, help="Per-channel sample rate in Hz.")
    parser.add_argument("--channels", type=int, default=2, help="Number of interleaved channels.")
    parser.add_argument("--segment-duration", type=float, default=2.0, help="Segment duration in seconds.")
    parser.add_argument("--time-resolution", type=float, default=0.002, help="Target spectrogram time step in seconds.")
    parser.add_argument(
        "--frequency-resolution",
        type=float,
        default=50.0,
        help="Target spectrogram frequency bin spacing in Hz.",
    )
    parser.add_argument("--freq-min", type=float, default=20000.0, help="Lower frequency bound in Hz.")
    parser.add_argument("--freq-max", type=float, default=30000.0, help="Upper frequency bound in Hz.")
    parser.add_argument("--aggregate", choices=("mean", "sum", "max"), default="mean", help="How to collapse power across frequency.")
    parser.add_argument("--threshold-mad", type=float, default=8.0, help="Threshold = median + K * robust_sigma.")
    parser.add_argument(
        "--min-peak-distance",
        type=float,
        default=0.01,
        help="Minimum spacing between auto-labeled peaks in seconds.",
    )
    parser.add_argument("--smooth-bins", type=int, default=3, help="Moving-average smoothing in time bins.")
    parser.add_argument(
        "--heatmap-sigma",
        type=float,
        default=0.01,
        help="Gaussian sigma in seconds for the target heatmap.",
    )
    parser.add_argument("--cmap", default="jet", help="Colormap for review PNGs.")
    parser.add_argument("--db-low", type=float, default=2.0, help="Lower percentile for PNG dB clipping.")
    parser.add_argument("--db-high", type=float, default=99.8, help="Upper percentile for PNG dB clipping.")
    parser.add_argument("--dpi", type=int, default=120, help="Review PNG DPI.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit the number of input files for a trial run.")
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Limit the total number of generated samples for a trial run.",
    )
    return parser.parse_args()


def collect_input_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    iterator = input_path.rglob(DEFAULT_PATTERN) if recursive else input_path.glob(DEFAULT_PATTERN)
    files = sorted(path for path in iterator if path.is_file())
    if not files:
        raise ValueError(f"No {DEFAULT_PATTERN} files found in {input_path}")
    return files


def resolve_channels(channel: str) -> list[tuple[str, int]]:
    if channel == "both":
        return [(name, CHANNEL_NAME_TO_INDEX[name]) for name in ("ns", "we")]
    return [(channel, CHANNEL_NAME_TO_INDEX[channel])]


def prepare_output_dirs(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    if output_dir.exists():
        if not overwrite and any(output_dir.iterdir()):
            raise ValueError(f"{output_dir} is not empty. Use --overwrite to rebuild the dataset.")
        if overwrite:
            shutil.rmtree(output_dir)

    samples_dir = output_dir / "samples"
    review_dir = output_dir / "review"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for stage in ("auto", "verified", "corrected"):
        for channel_name in ("ns", "we"):
            (review_dir / stage / channel_name).mkdir(parents=True, exist_ok=True)
    return samples_dir, review_dir


def compute_envelope_db(power: np.ndarray, aggregate: str, smooth_bins: int) -> np.ndarray:
    band_power = aggregate_band(power, aggregate)
    series_db = 10.0 * np.log10(np.maximum(band_power, 1e-12))
    return smooth_series(series_db, smooth_bins).astype(np.float32)


def build_target_heatmap(time_axis_s: np.ndarray, event_times_s: np.ndarray, sigma_s: float) -> np.ndarray:
    if sigma_s <= 0:
        raise ValueError("Heatmap sigma must be positive.")
    heatmap = np.zeros_like(time_axis_s, dtype=np.float32)
    if event_times_s.size == 0:
        return heatmap

    for event_time in event_times_s:
        gaussian = np.exp(-0.5 * ((time_axis_s - event_time) / sigma_s) ** 2)
        heatmap = np.maximum(heatmap, gaussian.astype(np.float32))
    return np.clip(heatmap, 0.0, 1.0)


def summarize_event_times(event_times_s: np.ndarray, max_items: int = 5) -> str:
    if event_times_s.size == 0:
        return "none"
    shown = [f"{time_s:.3f}" for time_s in event_times_s[:max_items]]
    suffix = "" if event_times_s.size <= max_items else ", ..."
    return ", ".join(shown) + suffix


def render_review_png(
    plt,
    spec_db: np.ndarray,
    time_axis_s: np.ndarray,
    freq_axis_hz: np.ndarray,
    output_path: Path,
    title: str,
    cmap: str,
    db_low: float,
    db_high: float,
    dpi: int,
) -> None:
    vmin, vmax = np.percentile(spec_db, [db_low, db_high])
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    image = ax.imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(time_axis_s[0], time_axis_s[-1], freq_axis_hz[0], freq_axis_hz[-1]),
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time, s")
    ax.set_ylabel("Frequency, Hz")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Power, dB")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_metadata(metadata_path: Path, rows: list[dict[str, object]]) -> None:
    with metadata_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METADATA_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    selected_channels = resolve_channels(args.channel)
    input_files = collect_input_files(input_path, recursive=args.recursive)
    if args.max_files is not None:
        input_files = input_files[: args.max_files]

    samples_dir, review_dir = prepare_output_dirs(output_dir, overwrite=args.overwrite)
    metadata_path = output_dir / "metadata.csv"
    plt = configure_matplotlib(show=False)

    metadata_rows: list[dict[str, object]] = []
    sample_counter = 0
    channel_counts = {name: 0 for name, _ in selected_channels}

    for file_index, file_path in enumerate(input_files, start=1):
        record_size, sample_count, data_offset = read_metadata(file_path)
        samples = load_samples(file_path, sample_count, record_size, data_offset)

        for channel_name, channel_index in selected_channels:
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

            for segment_index, start, stop, start_s, stop_s in segments:
                if args.max_segments is not None and sample_counter >= args.max_segments:
                    break

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
                time_axis_s = (time_axis / args.sample_rate).astype(np.float32)
                freq_axis_hz = (freq_axis * args.sample_rate).astype(np.float32)
                envelope_db = compute_envelope_db(power, args.aggregate, args.smooth_bins)
                _, _, threshold = robust_threshold(envelope_db, args.threshold_mad)
                peak_indices = pick_peaks(envelope_db, threshold=threshold, min_distance_bins=min_distance_bins)
                event_times_s = time_axis_s[peak_indices].astype(np.float32) if peak_indices.size else np.array([], dtype=np.float32)
                target_heatmap = build_target_heatmap(time_axis_s, event_times_s, sigma_s=args.heatmap_sigma)

                sample_id = f"loc_{sample_counter:07d}"
                npz_path = samples_dir / f"{sample_id}.npz"
                png_path = review_dir / "auto" / channel_name / f"{sample_id}.png"
                event_summary = summarize_event_times(event_times_s)
                title = (
                    f"{file_path.name} | {channel_name} | "
                    f"{start_s:.3f}-{stop_s:.3f} s | events={event_times_s.size}\n"
                    f"times: {event_summary}"
                )

                np.savez_compressed(
                    npz_path,
                    spec_db=spec_db.astype(np.float32),
                    envelope_db=envelope_db.astype(np.float32),
                    time_axis=time_axis_s.astype(np.float32),
                    freq_axis=freq_axis_hz.astype(np.float32),
                    source_file=np.array(str(file_path)),
                    channel=np.array(channel_name),
                    segment_start=np.array(start_s, dtype=np.float32),
                    segment_end=np.array(stop_s, dtype=np.float32),
                    event_times_s=event_times_s.astype(np.float32),
                    event_count=np.array(event_times_s.size, dtype=np.int32),
                    target_heatmap=target_heatmap.astype(np.float32),
                    label_source=np.array(DEFAULT_LABEL_SOURCE),
                    quality_flag=np.array(DEFAULT_QUALITY_FLAG),
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

                metadata_rows.append(
                    {
                        "sample_id": sample_id,
                        "source_file": str(file_path),
                        "channel": channel_name,
                        "segment_index": segment_index,
                        "segment_start": f"{start_s:.6f}",
                        "segment_end": f"{stop_s:.6f}",
                        "sample_rate": args.sample_rate,
                        "time_resolution": f"{hop / args.sample_rate:.6f}",
                        "frequency_resolution": f"{args.sample_rate / nperseg:.6f}",
                        "freq_min": f"{freq_min:.6f}",
                        "freq_max": f"{freq_max:.6f}",
                        "npz_path": str(npz_path.relative_to(output_dir)),
                        "png_path": str(png_path.relative_to(output_dir)),
                        "event_count": int(event_times_s.size),
                        "event_times_s_json": json.dumps([float(value) for value in event_times_s]),
                        "split": "",
                        "label_source": DEFAULT_LABEL_SOURCE,
                        "quality_flag": DEFAULT_QUALITY_FLAG,
                    }
                )
                channel_counts[channel_name] += 1
                sample_counter += 1

            if args.max_segments is not None and sample_counter >= args.max_segments:
                break

        print(
            f"[{file_index}/{len(input_files)}] {file_path.name}: "
            f"generated_samples={sample_counter}"
        )
        if args.max_segments is not None and sample_counter >= args.max_segments:
            break

    write_metadata(metadata_path, metadata_rows)
    print(f"Input files      : {len(input_files)}")
    print(f"Generated samples: {len(metadata_rows)}")
    print(f"Per channel      : {channel_counts}")
    print(f"NPZ samples      : {samples_dir}")
    print(f"Review PNGs      : {review_dir / 'auto'}")
    print(f"Metadata         : {metadata_path}")


if __name__ == "__main__":
    main()
