#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot_broadband_spectrogram import (
    CHANNEL_NAME_TO_INDEX,
    DEFAULT_SAMPLE_RATE,
    configure_matplotlib,
    extract_channel,
    load_samples,
    read_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read Broadband_Data *.bin, extract one ADC channel and save or "
            "display an amplitude-vs-time oscillogram in dBFS."
        )
    )
    parser.add_argument("input", type=Path, help="Path to Broadband_Data_*.bin")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PNG path. Defaults to <input>_<channel>_oscillogram.png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the oscillogram in a window instead of only saving it.",
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
        "--time-resolution",
        type=float,
        default=0.01,
        help="Time step for the envelope in seconds. Default: 0.01",
    )
    parser.add_argument(
        "--mode",
        choices=("rms", "peak"),
        default="rms",
        help="Envelope estimator in each time bin. Default: rms",
    )
    parser.add_argument(
        "--reference",
        type=float,
        default=32768.0,
        help="Reference amplitude for dB conversion. Default: 32768 (dBFS)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output image DPI. Default: 160",
    )
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


def compute_envelope_db(
    samples: np.ndarray,
    sample_rate: float,
    time_resolution: float,
    mode: str,
    reference: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if time_resolution <= 0:
        raise ValueError("Time resolution must be positive.")
    if reference <= 0:
        raise ValueError("Reference amplitude must be positive.")

    step = max(1, int(round(time_resolution * sample_rate)))
    usable = (samples.size // step) * step
    if usable == 0:
        raise ValueError("Time resolution is too large for the available samples.")

    trimmed = samples[:usable].reshape(-1, step)
    if mode == "rms":
        level = np.sqrt(np.mean(trimmed * trimmed, axis=1, dtype=np.float64))
    else:
        level = np.max(np.abs(trimmed), axis=1)

    envelope_db = 20.0 * np.log10(np.maximum(level / reference, 1e-12))
    time_axis = (np.arange(trimmed.shape[0], dtype=np.float64) + 0.5) * step / sample_rate
    return time_axis, envelope_db, step


def render_oscillogram(
    plt,
    time_axis: np.ndarray,
    envelope_db: np.ndarray,
    output: Path | None,
    input_path: Path,
    sample_rate: float,
    channel_name: str,
    mode: str,
    time_resolution: float,
    dpi: int,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.plot(time_axis, envelope_db, linewidth=1.0, color="#d94f04")
    ax.set_title(
        f"{input_path.name}\n"
        f"channel {channel_name}, {mode} envelope, sample rate {sample_rate:.0f} Hz, "
        f"time step {time_resolution:g} s"
    )
    ax.set_xlabel("Time, s")
    ax.set_ylabel("Amplitude, dBFS")
    ax.grid(True, alpha=0.25)

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
    input_path = args.input.expanduser().resolve()
    selected_channel, channel_index = resolve_channel(args)
    output_path = None
    if not args.no_save:
        output_path = (
            args.output.expanduser().resolve()
            if args.output
            else input_path.with_name(f"{input_path.stem}_{selected_channel}_oscillogram.png")
        )

    record_size, sample_count, data_offset = read_metadata(input_path)
    samples = load_samples(input_path, sample_count, record_size, data_offset)
    channel_samples = extract_channel(
        samples=samples,
        sample_count=sample_count,
        record_size=record_size,
        channels=args.channels,
        channel_index=channel_index,
    )
    time_axis, envelope_db, step = compute_envelope_db(
        samples=channel_samples,
        sample_rate=args.sample_rate,
        time_resolution=args.time_resolution,
        mode=args.mode,
        reference=args.reference,
    )
    render_oscillogram(
        plt=plt,
        time_axis=time_axis,
        envelope_db=envelope_db,
        output=output_path,
        input_path=input_path,
        sample_rate=args.sample_rate,
        channel_name=selected_channel,
        mode=args.mode,
        time_resolution=args.time_resolution,
        dpi=args.dpi,
        show=args.show,
    )

    print(f"Input file      : {input_path}")
    print(f"Record size     : {record_size}")
    print(f"Record count    : {sample_count}")
    print(f"Data offset     : {data_offset}")
    print(f"Channels        : {args.channels}")
    print(f"Channel         : {selected_channel}")
    print(f"Channel index   : {channel_index}")
    print(f"Channel samples : {channel_samples.size}")
    print(f"Sample rate     : {args.sample_rate}")
    print(f"Mode            : {args.mode}")
    print(f"Reference       : {args.reference}")
    print(f"Time step, s    : {step / args.sample_rate}")
    print(f"Duration, s     : {channel_samples.size / args.sample_rate}")
    print(f"Output image    : {output_path if output_path is not None else 'not saved'}")
    print(f"Show window     : {args.show}")


if __name__ == "__main__":
    main()
