#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from plot_broadband_spectrogram import (
    CHANNEL_NAME_TO_INDEX,
    compute_spectrogram,
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
            "Build a manually reviewable burst dataset from Broadband_Data *.bin files. "
            "Creates PNG spectrograms for labeling plus NPZ tensors for training."
        )
    )
    parser.add_argument("input", type=Path, help="Path to a .bin file or directory with .bin files.")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"), help="Dataset output directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory.")
    parser.add_argument("--sample-rate", type=float, default=100000.0, help="Per-channel sample rate in Hz.")
    parser.add_argument("--channels", type=int, default=2, help="Number of interleaved channels.")
    parser.add_argument("--channel", choices=sorted(CHANNEL_NAME_TO_INDEX), default="ns", help="Channel to extract.")
    parser.add_argument("--channel-index", type=int, default=None, help="Zero-based channel index override.")
    parser.add_argument("--segment-duration", type=float, default=2.0, help="Segment duration in seconds.")
    parser.add_argument("--time-resolution", type=float, default=0.002, help="Spectrogram time step in seconds.")
    parser.add_argument("--frequency-resolution", type=float, default=50.0, help="Spectrogram frequency bin spacing in Hz.")
    parser.add_argument("--freq-min", type=float, default=20000.0, help="Lower frequency bound in Hz.")
    parser.add_argument("--freq-max", type=float, default=30000.0, help="Upper frequency bound in Hz.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of input files for a trial run.")
    parser.add_argument("--max-segments", type=int, default=None, help="Limit total number of generated segments.")
    parser.add_argument("--cmap", default="jet", help="Colormap for review PNGs.")
    parser.add_argument("--db-low", type=float, default=2.0, help="Lower percentile for PNG dB clipping.")
    parser.add_argument("--db-high", type=float, default=99.8, help="Upper percentile for PNG dB clipping.")
    parser.add_argument("--dpi", type=int, default=120, help="Review PNG DPI.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset files.")
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


def prepare_output_dirs(output_dir: Path, overwrite: bool) -> tuple[Path, Path, Path, Path]:
    samples_dir = output_dir / "samples"
    review_dir = output_dir / "review"
    unlabeled_dir = review_dir / "unlabeled"
    burst_dir = review_dir / "burst"
    no_burst_dir = review_dir / "no_burst"
    metadata_path = output_dir / "metadata.csv"

    if metadata_path.exists() and not overwrite:
        raise ValueError(
            f"{metadata_path} already exists. Use --overwrite to rebuild the dataset."
        )

    samples_dir.mkdir(parents=True, exist_ok=True)
    unlabeled_dir.mkdir(parents=True, exist_ok=True)
    burst_dir.mkdir(parents=True, exist_ok=True)
    no_burst_dir.mkdir(parents=True, exist_ok=True)
    return samples_dir, unlabeled_dir, burst_dir, no_burst_dir


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


def compute_envelope_db(spec_db: np.ndarray) -> np.ndarray:
    power = np.power(10.0, spec_db / 10.0, dtype=np.float32)
    envelope = np.mean(power, axis=0)
    return 10.0 * np.log10(np.maximum(envelope, 1e-12))


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
    output_dir = args.output_dir.expanduser().resolve()
    samples_dir, unlabeled_dir, _, _ = prepare_output_dirs(output_dir, overwrite=args.overwrite)
    metadata_path = output_dir / "metadata.csv"
    selected_channel, channel_index = resolve_channel(args.channel, args.channel_index)
    input_files = collect_input_files(input_path, recursive=args.recursive)
    if args.max_files is not None:
        input_files = input_files[: args.max_files]

    plt = configure_matplotlib(show=False)
    metadata_rows: list[dict[str, object]] = []
    sample_counter = 0

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
            spec_db, time_axis, freq_axis = compute_spectrogram(segment_samples, nperseg=nperseg, hop=hop)
            spec_db, freq_axis = crop_frequency_band(
                spec_db=spec_db,
                freq_axis=freq_axis,
                sample_rate=args.sample_rate,
                freq_min=freq_min,
                freq_max=freq_max,
            )
            time_axis_s = time_axis / args.sample_rate
            freq_axis_hz = freq_axis * args.sample_rate
            envelope_db = compute_envelope_db(spec_db)

            sample_id = f"sample_{sample_counter:06d}"
            npz_path = samples_dir / f"{sample_id}.npz"
            png_path = unlabeled_dir / f"{sample_id}.png"
            title = (
                f"{sample_id} | {file_path.name} | {selected_channel} | "
                f"{start_s:.3f}-{stop_s:.3f} s | {freq_min:g}-{freq_max:g} Hz"
            )

            np.savez_compressed(
                npz_path,
                spec_db=spec_db.astype(np.float32),
                envelope_db=envelope_db.astype(np.float32),
                time_axis=time_axis_s.astype(np.float32),
                freq_axis=freq_axis_hz.astype(np.float32),
                label=np.array("unknown"),
                peak_time=np.array(np.nan, dtype=np.float32),
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
                    "time_resolution": args.time_resolution,
                    "frequency_resolution": args.frequency_resolution,
                    "freq_min": freq_min,
                    "freq_max": freq_max,
                    "npz_path": str(npz_path.relative_to(output_dir)),
                    "png_path": str(png_path.relative_to(output_dir)),
                    "label": "unknown",
                    "peak_time": "",
                    "split": "",
                    "label_source": "",
                }
            )
            sample_counter += 1

        print(
            f"{file_index}/{len(input_files)} {file_path.name}: "
            f"segments={len(segments)}, total_samples={sample_counter}"
        )
        if args.max_segments is not None and sample_counter >= args.max_segments:
            break

    write_metadata(metadata_path, metadata_rows)
    print(f"Input files      : {len(input_files)}")
    print(f"Generated samples: {len(metadata_rows)}")
    print(f"Review images    : {unlabeled_dir}")
    print(f"NPZ samples      : {samples_dir}")
    print(f"Metadata         : {metadata_path}")
    print("Manual labels    : move PNG files from review/unlabeled to review/burst or review/no_burst")


if __name__ == "__main__":
    main()
