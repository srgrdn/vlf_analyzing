#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

SIGNATURE = b"DTLG"
RECORD_SIZE_OFFSET = 0x08
SAMPLE_COUNT_OFFSET = 0x0242
HEADER_MIN_SIZE = SAMPLE_COUNT_OFFSET + 4
DEFAULT_SAMPLE_RATE = 100_000.0
CHANNEL_NAME_TO_INDEX = {"ns": 0, "we": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read Broadband_Data *.bin, extract the ADC sample stream and save a "
            "spectrogram image."
        )
    )
    parser.add_argument("input", type=Path, help="Path to Broadband_Data_*.bin")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PNG path. Defaults to <input>_spectrogram.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the spectrogram in a window instead of only saving it.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save PNG. Useful together with --show.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE,
        help="Per-channel sample rate in Hz. Default: 100000",
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=4096,
        help="FFT window length in samples for manual mode. Default: 4096",
    )
    parser.add_argument(
        "--time-resolution",
        type=float,
        default=None,
        help="Target time step between spectrogram columns in seconds.",
    )
    parser.add_argument(
        "--frequency-resolution",
        type=float,
        default=None,
        help="Target frequency bin spacing in Hz.",
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=None,
        help="Build consecutive spectrograms for fixed-length time segments in seconds.",
    )
    parser.add_argument(
        "--freq-min",
        type=float,
        default=None,
        help="Lower frequency bound in Hz for plotting.",
    )
    parser.add_argument(
        "--freq-max",
        type=float,
        default=None,
        help="Upper frequency bound in Hz for plotting.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=4000,
        help="Maximum number of time slices in automatic mode. Default: 4000",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output image DPI. Default: 160",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colormap for the spectrogram. Default: magma",
    )
    parser.add_argument(
        "--interpolation",
        choices=("nearest", "bilinear", "bicubic", "hanning", "lanczos"),
        default=None,
        help=(
            "Image interpolation for display. Default: hanning when --show is used, "
            "otherwise nearest."
        ),
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=2,
        help="Number of interleaved channels in each record. Default: 2",
    )
    parser.add_argument(
        "--channel",
        choices=sorted(CHANNEL_NAME_TO_INDEX),
        default="ns",
        help="Human-readable channel selector. Default: ns",
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        default=None,
        help="Zero-based channel index override. If omitted, --channel is used.",
    )
    parser.add_argument(
        "--db-low",
        type=float,
        default=2.0,
        help="Lower percentile for dB clipping. Default: 2",
    )
    parser.add_argument(
        "--db-high",
        type=float,
        default=99.8,
        help="Upper percentile for dB clipping. Default: 99.8",
    )
    return parser.parse_args()


def configure_matplotlib(show: bool):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def read_metadata(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as fh:
        header = fh.read(HEADER_MIN_SIZE)

    if len(header) < HEADER_MIN_SIZE:
        raise ValueError("File is too small to contain the expected header.")

    if header[:4] != SIGNATURE:
        raise ValueError("Unexpected file signature. Expected DTLG.")

    record_size = int.from_bytes(header[RECORD_SIZE_OFFSET : RECORD_SIZE_OFFSET + 4], "big")
    sample_count = int.from_bytes(header[SAMPLE_COUNT_OFFSET : SAMPLE_COUNT_OFFSET + 4], "big")
    payload_bytes = sample_count * record_size * 2
    file_size = path.stat().st_size
    data_offset = file_size - payload_bytes

    if record_size <= 0 or sample_count <= 0:
        raise ValueError("Parsed invalid metadata from the header.")
    if data_offset < 0:
        raise ValueError("Computed data offset is negative. Header values do not match file size.")

    return record_size, sample_count, data_offset


def load_samples(path: Path, sample_count: int, record_size: int, data_offset: int) -> np.ndarray:
    total_values = sample_count * record_size
    raw = np.memmap(path, dtype=">i2", mode="r", offset=data_offset, shape=(total_values,))
    return np.asarray(raw, dtype=np.float32)


def extract_channel(samples: np.ndarray, sample_count: int, record_size: int, channels: int, channel_index: int) -> np.ndarray:
    if channels <= 0:
        raise ValueError("Channel count must be positive.")
    if not 0 <= channel_index < channels:
        raise ValueError("Channel index is out of range.")
    if record_size % channels != 0:
        raise ValueError("Record size is not divisible by the requested channel count.")

    rows = samples.reshape(sample_count, record_size)
    return rows[:, channel_index::channels].reshape(-1)


def resolve_stft_parameters(
    samples: np.ndarray,
    sample_rate: float,
    nperseg: int,
    max_frames: int,
    time_resolution: float | None,
    frequency_resolution: float | None,
) -> tuple[int, int]:
    if time_resolution is not None and time_resolution <= 0:
        raise ValueError("Time resolution must be positive.")
    if frequency_resolution is not None and frequency_resolution <= 0:
        raise ValueError("Frequency resolution must be positive.")

    resolved_nperseg = nperseg
    if frequency_resolution is not None:
        resolved_nperseg = max(2, int(round(sample_rate / frequency_resolution)))

    if samples.size < resolved_nperseg:
        raise ValueError("Not enough samples for the requested FFT window.")

    if time_resolution is not None:
        hop = max(1, int(round(time_resolution * sample_rate)))
    else:
        hop = max(1, int(np.ceil((samples.size - resolved_nperseg) / max(max_frames - 1, 1))))

    return resolved_nperseg, hop


def resolve_frequency_band(
    sample_rate: float, freq_min: float | None, freq_max: float | None
) -> tuple[float, float]:
    nyquist = sample_rate / 2.0
    resolved_min = 0.0 if freq_min is None else freq_min
    resolved_max = nyquist if freq_max is None else freq_max

    if resolved_min < 0:
        raise ValueError("Minimum frequency must be non-negative.")
    if resolved_max > nyquist:
        raise ValueError(f"Maximum frequency cannot exceed Nyquist ({nyquist} Hz).")
    if resolved_min >= resolved_max:
        raise ValueError("Minimum frequency must be smaller than maximum frequency.")

    return resolved_min, resolved_max


def crop_frequency_band(
    spec_db: np.ndarray, freq_axis: np.ndarray, sample_rate: float, freq_min: float, freq_max: float
) -> tuple[np.ndarray, np.ndarray]:
    freq_hz = freq_axis * sample_rate
    mask = (freq_hz >= freq_min) & (freq_hz <= freq_max)
    if not np.any(mask):
        raise ValueError("The selected frequency band does not intersect the spectrum bins.")
    return spec_db[mask, :], freq_axis[mask]


def iter_time_segments(
    samples: np.ndarray, sample_rate: float, segment_duration: float | None, nperseg: int
) -> list[tuple[int, int, int, float, float]]:
    if segment_duration is None:
        return [(0, 0, samples.size, 0.0, samples.size / sample_rate)]
    if segment_duration <= 0:
        raise ValueError("Segment duration must be positive.")

    segment_samples = max(1, int(round(segment_duration * sample_rate)))
    segments: list[tuple[int, int, int, float, float]] = []
    start = 0
    index = 0
    while start < samples.size:
        stop = min(start + segment_samples, samples.size)
        if stop - start < nperseg:
            break
        segments.append((index, start, stop, start / sample_rate, stop / sample_rate))
        start = stop
        index += 1
    if not segments:
        raise ValueError("Segment duration is too short or the file is too small for the chosen FFT window.")
    return segments


def resolve_output_path(
    output_path: Path | None,
    input_path: Path,
    segment_index: int,
    segment_start_s: float,
    segment_stop_s: float,
    segmented: bool,
) -> Path | None:
    if output_path is None:
        return None

    base = output_path

    if not segmented:
        return base

    suffix = base.suffix or ".png"
    stem = base.stem if base.suffix else base.name
    name = (
        f"{stem}_seg{segment_index:03d}_"
        f"{segment_start_s:07.3f}-{segment_stop_s:07.3f}s{suffix}"
    )
    return base.with_name(name)


def compute_spectrogram(
    samples: np.ndarray, nperseg: int, hop: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    spec_db = 10.0 * np.log10(np.maximum(power, 1e-12))

    time_centers = (np.arange(frame_count, dtype=np.float32) * hop + nperseg / 2.0).astype(np.float32)
    freq_bins = np.fft.rfftfreq(nperseg, d=1.0)
    return spec_db.T, time_centers, freq_bins


def render_spectrogram(
    plt,
    spec_db: np.ndarray,
    time_axis: np.ndarray,
    freq_axis: np.ndarray,
    output: Path | None,
    input_path: Path,
    sample_rate: float,
    sample_count: int,
    record_size: int,
    channel_samples: int,
    cmap: str,
    dpi: int,
    db_low: float,
    db_high: float,
    show: bool,
    hop: int,
    interpolation: str,
    segment_label: str | None,
    freq_min: float,
    freq_max: float,
    nperseg: int,
) -> None:
    vmin, vmax = np.percentile(spec_db, [db_low, db_high])
    x_axis = time_axis / sample_rate
    y_axis = freq_axis * sample_rate
    xlabel = "Time, s"
    ylabel = "Frequency, Hz"

    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    image = ax.imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]),
        interpolation=interpolation,
    )

    ax.set_title(
        f"{input_path.name}\n"
        f"{sample_count} records x {record_size} values, channel stream {channel_samples} samples, "
        f"window {nperseg}, hop {hop}, band {freq_min:g}-{freq_max:g} Hz"
    )
    if segment_label:
        ax.set_title(f"{ax.get_title()}\n{segment_label}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Power, dB")

    if output is not None:
        fig.savefig(output, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.no_save and not args.show:
        raise ValueError("Nothing to do: use --show, or omit --no-save to save PNG.")

    plt = configure_matplotlib(show=args.show)
    interpolation = args.interpolation or ("hanning" if args.show else "nearest")
    input_path = args.input.expanduser().resolve()
    output_path = None
    if not args.no_save:
        output_path = (
            args.output.expanduser().resolve()
            if args.output
            else input_path.with_name(f"{input_path.stem}_spectrogram.png")
        )

    record_size, sample_count, data_offset = read_metadata(input_path)
    samples = load_samples(input_path, sample_count, record_size, data_offset)
    selected_channel = args.channel
    channel_index = args.channel_index
    if channel_index is None:
        channel_index = CHANNEL_NAME_TO_INDEX[selected_channel]
    else:
        selected_channel = next(
            (name for name, idx in CHANNEL_NAME_TO_INDEX.items() if idx == channel_index),
            f"index-{channel_index}",
        )
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
    freq_min, freq_max = resolve_frequency_band(
        sample_rate=args.sample_rate,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
    )
    segments = iter_time_segments(
        samples=channel_samples,
        sample_rate=args.sample_rate,
        segment_duration=args.segment_duration,
        nperseg=nperseg,
    )

    for segment_index, start, stop, start_s, stop_s in segments:
        segment_samples = channel_samples[start:stop]
        spec_db, time_axis, freq_axis = compute_spectrogram(
            samples=segment_samples,
            nperseg=nperseg,
            hop=hop,
        )
        spec_db, freq_axis = crop_frequency_band(
            spec_db=spec_db,
            freq_axis=freq_axis,
            sample_rate=args.sample_rate,
            freq_min=freq_min,
            freq_max=freq_max,
        )
        segment_time_axis = time_axis + start
        segment_label = None
        if args.segment_duration is not None:
            segment_label = (
                f"segment {segment_index + 1}/{len(segments)}: "
                f"{start_s:.3f}-{stop_s:.3f} s"
            )
        segment_output = resolve_output_path(
            output_path=output_path,
            input_path=input_path,
            segment_index=segment_index,
            segment_start_s=start_s,
            segment_stop_s=stop_s,
            segmented=args.segment_duration is not None,
        )
        render_spectrogram(
            plt=plt,
            spec_db=spec_db,
            time_axis=segment_time_axis,
            freq_axis=freq_axis,
            output=segment_output,
            input_path=input_path,
            sample_rate=args.sample_rate,
            sample_count=sample_count,
            record_size=record_size,
            channel_samples=channel_samples.size,
            cmap=args.cmap,
            dpi=args.dpi,
            db_low=args.db_low,
            db_high=args.db_high,
            show=args.show,
            hop=hop,
            interpolation=interpolation,
            segment_label=segment_label,
            freq_min=freq_min,
            freq_max=freq_max,
            nperseg=nperseg,
        )
        print(
            f"Built segment    : {segment_index + 1}/{len(segments)} "
            f"({start_s:.3f}-{stop_s:.3f} s) -> "
            f"{segment_output if segment_output is not None else 'not saved'}"
        )

    duration_seconds = channel_samples.size / args.sample_rate
    max_frequency_hz = args.sample_rate / 2.0
    time_step_seconds = hop / args.sample_rate
    frequency_step_hz = args.sample_rate / nperseg

    print(f"Input file      : {input_path}")
    print(f"Record size     : {record_size}")
    print(f"Record count    : {sample_count}")
    print(f"Data offset     : {data_offset}")
    print(f"Channels        : {args.channels}")
    print(f"Channel         : {selected_channel}")
    print(f"Channel index   : {channel_index}")
    print(f"Channel samples : {channel_samples.size}")
    print(f"Sample rate     : {args.sample_rate}")
    print(f"Window samples  : {nperseg}")
    print(f"Hop samples     : {hop}")
    print(f"Time step, s    : {time_step_seconds}")
    print(f"Freq step, Hz   : {frequency_step_hz}")
    print(f"Duration, s     : {duration_seconds}")
    print(f"Freq range, Hz  : 0..{max_frequency_hz}")
    print(f"Plot band, Hz   : {freq_min}..{freq_max}")
    print(f"Segments        : {len(segments)}")
    if args.segment_duration is not None:
        print(f"Segment size, s : {args.segment_duration}")
    print(f"Output image    : {output_path if output_path is not None else 'not saved'}")
    print(f"Show window     : {args.show}")
    print(f"Colormap        : {args.cmap}")
    print(f"Interpolation   : {interpolation}")


if __name__ == "__main__":
    main()
