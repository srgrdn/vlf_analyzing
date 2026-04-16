#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

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
            "Detect short broadband bursts by aggregating spectrogram power in a selected "
            "frequency band and applying a robust peak threshold."
        )
    )
    parser.add_argument("input", type=Path, help="Path to Broadband_Data_*.bin")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Base PNG path for saved segment plots. Default: <input>_burst_detection.png",
    )
    parser.add_argument("--show", action="store_true", help="Display plots in a window.")
    parser.add_argument("--no-save", action="store_true", help="Do not save PNG or CSV results.")
    parser.add_argument("--sample-rate", type=float, default=100000.0, help="Per-channel sample rate in Hz.")
    parser.add_argument("--nperseg", type=int, default=4096, help="FFT window length for manual mode.")
    parser.add_argument("--time-resolution", type=float, default=0.002, help="Target time step in seconds.")
    parser.add_argument("--frequency-resolution", type=float, default=50.0, help="Target frequency bin spacing in Hz.")
    parser.add_argument("--max-frames", type=int, default=4000, help="Maximum time slices in automatic mode.")
    parser.add_argument("--channels", type=int, default=2, help="Number of interleaved channels.")
    parser.add_argument("--channel", choices=sorted(CHANNEL_NAME_TO_INDEX), default="ns", help="Human-readable channel selector.")
    parser.add_argument("--channel-index", type=int, default=None, help="Zero-based channel index override.")
    parser.add_argument("--freq-min", type=float, required=True, help="Lower frequency bound in Hz.")
    parser.add_argument("--freq-max", type=float, required=True, help="Upper frequency bound in Hz.")
    parser.add_argument("--segment-duration", type=float, default=2.0, help="Segment duration in seconds.")
    parser.add_argument("--aggregate", choices=("mean", "sum", "max"), default="mean", help="How to collapse the selected band across frequency.")
    parser.add_argument("--threshold-mad", type=float, default=6.0, help="Threshold = median + K * robust_sigma, where robust_sigma = 1.4826 * MAD.")
    parser.add_argument("--min-peak-distance", type=float, default=0.01, help="Minimum spacing between detections in seconds.")
    parser.add_argument("--smooth-bins", type=int, default=3, help="Moving-average smoothing in time bins. Default: 3")
    parser.add_argument("--cmap", default="jet", help="Colormap for the spectrogram panel. Default: jet")
    parser.add_argument("--db-low", type=float, default=2.0, help="Lower percentile for spectrogram dB clipping.")
    parser.add_argument("--db-high", type=float, default=99.8, help="Upper percentile for spectrogram dB clipping.")
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI.")
    return parser.parse_args()


def resolve_channel(args: argparse.Namespace) -> tuple[str, int]:
    channel_index = args.channel_index
    selected_channel = args.channel
    if channel_index is None:
        channel_index = CHANNEL_NAME_TO_INDEX[selected_channel]
    else:
        selected_channel = next(
            (name for name, idx in CHANNEL_NAME_TO_INDEX.items() if idx == channel_index),
            f"index-{channel_index}",
        )
    return selected_channel, channel_index


def compute_spectrogram_power(samples: np.ndarray, nperseg: int, hop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if samples.size < nperseg:
        raise ValueError("Not enough samples for the requested FFT window.")

    frame_count = 1 + (samples.size - nperseg) // hop
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(frame_count, nperseg),
        strides=(samples.strides[0] * hop, samples.strides[0]),
        writeable=False,
    )
    window = np.hanning(nperseg).astype(np.float32)
    centered = frames - frames.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered * window, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    power /= np.sum(window**2, dtype=np.float32)

    time_centers = (np.arange(frame_count, dtype=np.float32) * hop + nperseg / 2.0).astype(np.float32)
    freq_bins = np.fft.rfftfreq(nperseg, d=1.0)
    return power.T, time_centers, freq_bins


def aggregate_band(power: np.ndarray, aggregate: str) -> np.ndarray:
    if aggregate == "mean":
        return np.mean(power, axis=0)
    if aggregate == "sum":
        return np.sum(power, axis=0)
    return np.max(power, axis=0)


def smooth_series(values: np.ndarray, smooth_bins: int) -> np.ndarray:
    if smooth_bins <= 1:
        return values
    kernel = np.ones(smooth_bins, dtype=np.float32) / smooth_bins
    return np.convolve(values, kernel, mode="same")


def robust_threshold(series_db: np.ndarray, threshold_mad: float) -> tuple[float, float, float]:
    median = float(np.median(series_db))
    mad = float(np.median(np.abs(series_db - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma == 0.0:
        robust_sigma = float(np.std(series_db))
    threshold = median + threshold_mad * robust_sigma
    return median, robust_sigma, threshold


def pick_peaks(series_db: np.ndarray, threshold: float, min_distance_bins: int) -> np.ndarray:
    if series_db.size < 3:
        return np.array([], dtype=int)

    candidates = np.flatnonzero(
        (series_db[1:-1] >= series_db[:-2]) &
        (series_db[1:-1] > series_db[2:]) &
        (series_db[1:-1] >= threshold)
    ) + 1
    if candidates.size == 0:
        return candidates

    order = candidates[np.argsort(series_db[candidates])[::-1]]
    selected: list[int] = []
    for idx in order:
        if all(abs(idx - prev) >= min_distance_bins for prev in selected):
            selected.append(int(idx))
    return np.array(sorted(selected), dtype=int)


def resolve_output_base(input_path: Path, output: Path | None, save_enabled: bool) -> Path | None:
    if not save_enabled:
        return None
    if output is not None:
        return output.resolve()
    return input_path.with_name(f"{input_path.stem}_burst_detection.png")


def resolve_segment_output(base: Path | None, segment_index: int, start_s: float, stop_s: float) -> Path | None:
    if base is None:
        return None
    suffix = base.suffix or ".png"
    stem = base.stem if base.suffix else base.name
    return base.with_name(
        f"{stem}_seg{segment_index:03d}_{start_s:07.3f}-{stop_s:07.3f}s{suffix}"
    )


def render_detection_plot(
    plt,
    spec_db: np.ndarray,
    time_s: np.ndarray,
    freq_hz: np.ndarray,
    series_db: np.ndarray,
    threshold: float,
    peak_indices: np.ndarray,
    output: Path | None,
    input_path: Path,
    channel_name: str,
    freq_min: float,
    freq_max: float,
    segment_label: str,
    aggregate: str,
    cmap: str,
    db_low: float,
    db_high: float,
    dpi: int,
    show: bool,
) -> None:
    vmin, vmax = np.percentile(spec_db, [db_low, db_high])
    fig, (ax_spec, ax_env) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        constrained_layout=True,
        sharex=True,
        height_ratios=(3, 2),
    )
    image = ax_spec.imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(time_s[0], time_s[-1], freq_hz[0], freq_hz[-1]),
        interpolation="nearest",
    )
    if peak_indices.size:
        ax_spec.vlines(
            time_s[peak_indices],
            ymin=freq_hz[0],
            ymax=freq_hz[-1],
            color="white",
            linewidth=1.0,
            alpha=0.9,
        )
    ax_spec.set_ylabel("Frequency, Hz")
    ax_spec.set_title(
        f"{input_path.name}\n"
        f"{segment_label}, channel {channel_name}, band {freq_min:g}-{freq_max:g} Hz, aggregate {aggregate}"
    )
    cbar = fig.colorbar(image, ax=ax_spec)
    cbar.set_label("Power, dB")

    ax_env.plot(time_s, series_db, color="#1d4ed8", linewidth=1.1, label="Band envelope")
    ax_env.axhline(threshold, color="#dc2626", linestyle="--", linewidth=1.3, label="Threshold")
    if peak_indices.size:
        ax_env.scatter(
            time_s[peak_indices],
            series_db[peak_indices],
            color="#ea580c",
            s=32,
            zorder=3,
            label="Detections",
        )
    ax_env.set_xlabel("Time, s")
    ax_env.set_ylabel("Band power, dB")
    ax_env.grid(True, alpha=0.25)
    ax_env.legend()

    if output is not None:
        fig.savefig(output, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def write_detections_csv(rows: list[dict[str, object]], output_path: Path) -> Path:
    fieldnames = [
        "segment_index",
        "segment_start_s",
        "segment_stop_s",
        "peak_time_global_s",
        "peak_time_local_s",
        "peak_db",
        "threshold_db",
    ]
    candidates = [output_path]
    if output_path.suffix:
        candidates.append(output_path.with_name(f"{output_path.stem}_new{output_path.suffix}"))
    else:
        candidates.append(output_path.with_name(f"{output_path.name}_new"))

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            if candidate.exists():
                os.chmod(candidate, 0o666)
            with candidate.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            return candidate
        except PermissionError as exc:
            last_error = exc
            continue

    assert last_error is not None
    raise last_error


def main() -> None:
    args = parse_args()
    if args.no_save and not args.show:
        raise ValueError("Nothing to do: use --show, or omit --no-save to save results.")

    input_path = args.input.expanduser().resolve()
    selected_channel, channel_index = resolve_channel(args)
    plt = configure_matplotlib(show=args.show)

    record_size, sample_count, data_offset = read_metadata(input_path)
    samples = load_samples(input_path, sample_count, record_size, data_offset)
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
        nperseg=args.nperseg,
        max_frames=args.max_frames,
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

    base_output = resolve_output_base(input_path, args.output, save_enabled=not args.no_save)
    detection_rows: list[dict[str, object]] = []
    min_distance_bins = max(1, int(round(args.min_peak_distance * args.sample_rate / hop)))

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
        spec_db = 10.0 * np.log10(np.maximum(power, 1e-12))
        freq_hz = freq_axis * args.sample_rate

        band_power = aggregate_band(power, args.aggregate)
        series_db = 10.0 * np.log10(np.maximum(band_power, 1e-12))
        series_db = smooth_series(series_db, args.smooth_bins)

        _, _, threshold = robust_threshold(series_db, args.threshold_mad)
        peak_indices = pick_peaks(series_db, threshold=threshold, min_distance_bins=min_distance_bins)

        local_time_s = time_axis / args.sample_rate
        global_time_s = local_time_s + start_s
        segment_label = f"segment {segment_index + 1}/{len(segments)}: {start_s:.3f}-{stop_s:.3f} s"
        segment_output = resolve_segment_output(base_output, segment_index, start_s, stop_s)

        render_detection_plot(
            plt=plt,
            spec_db=spec_db,
            time_s=global_time_s,
            freq_hz=freq_hz,
            series_db=series_db,
            threshold=threshold,
            peak_indices=peak_indices,
            output=segment_output,
            input_path=input_path,
            channel_name=selected_channel,
            freq_min=freq_min,
            freq_max=freq_max,
            segment_label=segment_label,
            aggregate=args.aggregate,
            cmap=args.cmap,
            db_low=args.db_low,
            db_high=args.db_high,
            dpi=args.dpi,
            show=args.show,
        )

        for idx in peak_indices:
            detection_rows.append(
                {
                    "segment_index": segment_index,
                    "segment_start_s": f"{start_s:.6f}",
                    "segment_stop_s": f"{stop_s:.6f}",
                    "peak_time_global_s": f"{global_time_s[idx]:.6f}",
                    "peak_time_local_s": f"{local_time_s[idx]:.6f}",
                    "peak_db": f"{series_db[idx]:.6f}",
                    "threshold_db": f"{threshold:.6f}",
                }
            )

        print(
            f"Segment {segment_index + 1:02d}/{len(segments)} "
            f"{start_s:.3f}-{stop_s:.3f} s: detections={len(peak_indices)} "
            f"threshold={threshold:.2f} dB"
        )

    csv_output = None if base_output is None else base_output.with_name(f"{base_output.stem}_detections.csv")
    if csv_output is not None:
        csv_output = write_detections_csv(detection_rows, csv_output)

    print(f"Input file        : {input_path}")
    print(f"Channel           : {selected_channel}")
    print(f"Sample rate       : {args.sample_rate}")
    print(f"Time step, s      : {hop / args.sample_rate}")
    print(f"Freq step, Hz     : {args.sample_rate / nperseg}")
    print(f"Band, Hz          : {freq_min}..{freq_max}")
    print(f"Segments          : {len(segments)}")
    print(f"Total detections  : {len(detection_rows)}")
    print(f"Aggregate         : {args.aggregate}")
    print(f"Threshold MAD     : {args.threshold_mad}")
    print(f"Min peak dist, s  : {args.min_peak_distance}")
    print(f"Output base       : {base_output if base_output is not None else 'not saved'}")
    print(f"CSV output        : {csv_output if csv_output is not None else 'not saved'}")


if __name__ == "__main__":
    main()
