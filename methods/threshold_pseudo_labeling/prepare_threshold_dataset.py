from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np


def find_project_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "detect_broadband_bursts.py").exists() and (candidate / "plot_broadband_spectrogram.py").exists():
            return candidate
    raise FileNotFoundError("Could not find project root with detector scripts.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_burst_dataset import render_review_png  # noqa: E402
from detect_broadband_bursts import (  # noqa: E402
    aggregate_band,
    compute_spectrogram_power,
    pick_peaks,
    robust_threshold,
    smooth_series,
)
from plot_broadband_spectrogram import (  # noqa: E402
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


LABELS = ("no_burst", "burst")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a pseudo-labeled training dataset using detect_broadband_bursts.py logic."
    )
    parser.add_argument("input", type=Path, help="Path to a .bin file or directory with .bin files.")
    parser.add_argument("--output-dir", type=Path, default=Path("methods/threshold_pseudo_labeling/dataset"))
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory.")
    parser.add_argument("--sample-rate", type=float, default=100000.0)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--channel", choices=sorted(CHANNEL_NAME_TO_INDEX), default="ns")
    parser.add_argument("--channel-index", type=int, default=None)
    parser.add_argument("--segment-duration", type=float, default=2.0)
    parser.add_argument("--time-resolution", type=float, default=0.002)
    parser.add_argument("--frequency-resolution", type=float, default=50.0)
    parser.add_argument("--freq-min", type=float, default=20000.0)
    parser.add_argument("--freq-max", type=float, default=30000.0)
    parser.add_argument("--aggregate", choices=("mean", "sum", "max"), default="mean")
    parser.add_argument("--threshold-mad", type=float, default=8.0)
    parser.add_argument("--min-peak-distance", type=float, default=0.01)
    parser.add_argument("--smooth-bins", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--db-low", type=float, default=2.0)
    parser.add_argument("--db-high", type=float, default=99.8)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", default="threshold-pseudo-v1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_input_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")
    iterator = input_path.rglob("*.bin") if recursive else input_path.glob("*.bin")
    files = sorted(path for path in iterator if path.is_file())
    if not files:
        raise ValueError(f"No .bin files found in {input_path}")
    return files


def resolve_channel(channel: str, channel_index: int | None) -> tuple[str, int]:
    if channel_index is None:
        return channel, CHANNEL_NAME_TO_INDEX[channel]
    selected_channel = next(
        (name for name, idx in CHANNEL_NAME_TO_INDEX.items() if idx == channel_index),
        f"index-{channel_index}",
    )
    return selected_channel, channel_index


def split_for_sample(sample_id: str, label: str, train_ratio: float, val_ratio: float, seed: str) -> str:
    digest = hashlib.sha256(f"{seed}:{label}:{sample_id}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def prepare_output_dirs(output_dir: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    metadata_path = output_dir / "metadata.csv"
    if metadata_path.exists() and not overwrite:
        raise ValueError(f"{metadata_path} already exists. Use --overwrite to rebuild.")

    samples_dir = output_dir / "samples"
    review_dir = output_dir / "review"
    verify_dir = output_dir / "verify"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for label in LABELS:
        (review_dir / label).mkdir(parents=True, exist_ok=True)
        (verify_dir / label).mkdir(parents=True, exist_ok=True)
    return samples_dir, review_dir, verify_dir


def compute_envelope_db(power: np.ndarray, aggregate: str, smooth_bins: int) -> np.ndarray:
    band_power = aggregate_band(power, aggregate)
    series_db = 10.0 * np.log10(np.maximum(band_power, 1e-12))
    return smooth_series(series_db, smooth_bins).astype(np.float32)


def write_metadata(metadata_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
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
        "label",
        "peak_time",
        "peak_count",
        "threshold_db",
        "split",
        "label_source",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()

    samples_dir, review_dir, verify_dir = prepare_output_dirs(output_dir, args.overwrite)
    metadata_path = output_dir / "metadata.csv"
    selected_channel, channel_index = resolve_channel(args.channel, args.channel_index)
    input_files = collect_input_files(input_path, args.recursive)
    if args.max_files is not None:
        input_files = input_files[: args.max_files]

    plt = configure_matplotlib(show=False)
    metadata_rows: list[dict[str, object]] = []
    sample_counter = 0
    label_counts = {"burst": 0, "no_burst": 0}
    split_counts: dict[tuple[str, str], int] = {}

    for file_index, file_path in enumerate(input_files, start=1):
        record_size, sample_count, data_offset = read_metadata(file_path)
        samples = load_samples(file_path, sample_count, record_size, data_offset)
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

        print(f"[{file_index}/{len(input_files)}] {file_path.name}: {len(segments)} segments")
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
            envelope_db = compute_envelope_db(power, args.aggregate, args.smooth_bins)
            _, _, threshold = robust_threshold(envelope_db, args.threshold_mad)
            peak_indices = pick_peaks(envelope_db, threshold=threshold, min_distance_bins=min_distance_bins)

            local_time_s = time_axis / args.sample_rate
            time_axis_s = local_time_s.astype(np.float32)
            freq_axis_hz = (freq_axis * args.sample_rate).astype(np.float32)
            peak_times = local_time_s[peak_indices] + start_s if peak_indices.size else np.array([], dtype=np.float32)
            peak_time = float(peak_times[0]) if peak_times.size else np.nan
            label = "burst" if peak_indices.size else "no_burst"
            sample_id = f"pseudo_{sample_counter:06d}"
            split = split_for_sample(sample_id, label, args.train_ratio, args.val_ratio, args.split_seed)

            npz_path = samples_dir / f"{sample_id}.npz"
            png_path = review_dir / label / f"{sample_id}.png"
            title = (
                f"{sample_id} | {file_path.name} | detector={label} | "
                f"peaks={len(peak_indices)} | {start_s:.3f}-{stop_s:.3f} s"
            )

            np.savez_compressed(
                npz_path,
                spec_db=spec_db.astype(np.float32),
                envelope_db=envelope_db.astype(np.float32),
                time_axis=time_axis_s,
                freq_axis=freq_axis_hz,
                label=np.array(label),
                peak_time=np.array(peak_time, dtype=np.float32),
                peak_count=np.array(len(peak_indices), dtype=np.int32),
                threshold_db=np.array(threshold, dtype=np.float32),
                source_file=np.array(str(file_path)),
                channel=np.array(selected_channel),
                segment_start=np.array(start_s, dtype=np.float32),
                segment_end=np.array(stop_s, dtype=np.float32),
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
                    "channel": selected_channel,
                    "segment_index": segment_index,
                    "segment_start": f"{start_s:.6f}",
                    "segment_end": f"{stop_s:.6f}",
                    "sample_rate": args.sample_rate,
                    "time_resolution": hop / args.sample_rate,
                    "frequency_resolution": args.sample_rate / nperseg,
                    "freq_min": freq_min,
                    "freq_max": freq_max,
                    "npz_path": str(npz_path.relative_to(output_dir)),
                    "png_path": str(png_path.relative_to(output_dir)),
                    "label": label,
                    "peak_time": "" if np.isnan(peak_time) else f"{peak_time:.6f}",
                    "peak_count": len(peak_indices),
                    "threshold_db": f"{threshold:.6f}",
                    "split": split,
                    "label_source": "threshold_detector",
                }
            )
            label_counts[label] += 1
            split_counts[(split, label)] = split_counts.get((split, label), 0) + 1
            sample_counter += 1

        if args.max_segments is not None and sample_counter >= args.max_segments:
            break

    write_metadata(metadata_path, metadata_rows)
    print(f"Output dataset : {output_dir}")
    print(f"Metadata       : {metadata_path}")
    print(f"Verify dirs    : {verify_dir / 'burst'} | {verify_dir / 'no_burst'}")
    print(f"Samples        : {sample_counter}")
    print(f"Labels         : {label_counts}")
    print(f"Splits         : {dict(sorted(split_counts.items()))}")


if __name__ == "__main__":
    main()
