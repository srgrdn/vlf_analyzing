#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


METHOD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = METHOD_DIR.parent.parent
DEFAULT_MODEL = METHOD_DIR / "models" / "sferics_localization_2d.pt"
DEFAULT_OUTPUT_DIR = METHOD_DIR / "reports" / "raw_inference"
DEFAULT_PATTERN = "Broadband_Data_*.bin"
CHANNEL_CHOICES = ("ns", "we", "both")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the saved 2D sferics localization CNN on raw Broadband_Data *.bin "
            "files. The script builds 2-second spectrogram windows, predicts temporal "
            "heatmaps, peak-picks event times, and writes CSV/JSON summaries."
        )
    )
    parser.add_argument("input", type=Path, help="Broadband_Data_*.bin file or directory.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Model checkpoint path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for CSV/JSON and optional PNGs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob pattern when input is a directory. Default: {DEFAULT_PATTERN}",
    )
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively.")
    parser.add_argument(
        "--channel",
        choices=CHANNEL_CHOICES,
        default="both",
        help="Channel to process. Default: both",
    )
    parser.add_argument("--sample-rate", type=float, default=100_000.0, help="Per-channel sample rate in Hz.")
    parser.add_argument("--channels", type=int, default=2, help="Number of interleaved channels. Default: 2")
    parser.add_argument("--segment-duration", type=float, default=2.0, help="Segment duration in seconds.")
    parser.add_argument("--time-resolution", type=float, default=0.002, help="Target spectrogram time step in seconds.")
    parser.add_argument("--frequency-resolution", type=float, default=50.0, help="Target frequency step in Hz.")
    parser.add_argument("--freq-min", type=float, default=20_000.0, help="Lower frequency bound in Hz.")
    parser.add_argument("--freq-max", type=float, default=30_000.0, help="Upper frequency bound in Hz.")
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=0.35,
        help="Predicted heatmap peak threshold used to convert heatmap peaks to event times.",
    )
    parser.add_argument(
        "--min-peak-distance",
        type=float,
        default=0.03,
        help="Minimum distance between predicted events in seconds.",
    )
    parser.add_argument("--nperseg", type=int, default=4096, help="Fallback FFT window length.")
    parser.add_argument("--max-frames", type=int, default=4000, help="Maximum frames for automatic hop fallback.")
    parser.add_argument("--db-low", type=float, default=2.0, help="Lower percentile for plot clipping.")
    parser.add_argument("--db-high", type=float, default=99.8, help="Upper percentile for plot clipping.")
    parser.add_argument("--cmap", default="jet", help="Spectrogram colormap for plots. Default: jet")
    parser.add_argument("--dpi", type=int, default=160, help="Plot DPI. Default: 160")
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save spectrogram+heatmap PNGs for selected segments.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show selected spectrogram+heatmap plots interactively.",
    )
    parser.add_argument(
        "--plot-top",
        type=int,
        default=6,
        help="Plot top-N most active segments per channel/file when --save-plots or --show is used. Default: 6",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Plot every processed segment when --save-plots or --show is used.",
    )
    return parser.parse_args()


def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_matplotlib(show: bool):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def resolve_input_files(input_path: Path, pattern: str, recursive: bool) -> list[Path]:
    resolved = input_path.expanduser().resolve()
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"Input path not found: {resolved}")

    globber = resolved.rglob if recursive else resolved.glob
    files = sorted(path for path in globber(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {resolved}")
    return files


def selected_channels(channel_arg: str) -> list[str]:
    return ["ns", "we"] if channel_arg == "both" else [channel_arg]


def output_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def predict_segment(model, torch_mod, device, loc, spec_db: np.ndarray) -> np.ndarray:
    normalized = loc.normalize_spec(spec_db)
    with torch_mod.no_grad():
        prediction = model(torch_mod.from_numpy(normalized[None, None, :, :]).to(device))
    return prediction.detach().cpu().numpy()[0].astype(np.float32)


def extract_predicted_times(
    loc,
    prediction: np.ndarray,
    time_axis_s: np.ndarray,
    peak_threshold: float,
    min_peak_distance_s: float,
) -> list[float]:
    time_step = float(np.median(np.diff(time_axis_s))) if time_axis_s.size > 1 else 0.002
    min_distance_bins = max(1, int(round(min_peak_distance_s / time_step)))
    peak_indices = loc.pick_peaks(
        prediction,
        threshold=peak_threshold,
        min_distance_bins=min_distance_bins,
    )
    return [float(time_axis_s[index]) for index in peak_indices]


def render_prediction_plot(
    plt,
    *,
    output: Path | None,
    show: bool,
    source_file: Path,
    channel: str,
    segment_index: int,
    segment_start_s: float,
    segment_end_s: float,
    spec_db: np.ndarray,
    time_axis_s: np.ndarray,
    freq_axis_hz: np.ndarray,
    prediction: np.ndarray,
    predicted_times_inside: list[float],
    peak_threshold: float,
    db_low: float,
    db_high: float,
    cmap: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    vmin, vmax = np.percentile(spec_db, [db_low, db_high])
    axes[0].imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        extent=(float(time_axis_s[0]), float(time_axis_s[-1]), float(freq_axis_hz[0]), float(freq_axis_hz[-1])),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    for event_time in predicted_times_inside:
        axes[0].axvline(event_time, color="white", linewidth=1.0, alpha=0.9)
    axes[0].set_title(
        f"{source_file.name} | {channel} | segment {segment_index:03d} | "
        f"{segment_start_s:.3f}-{segment_end_s:.3f}s | predicted={len(predicted_times_inside)}"
    )
    axes[0].set_ylabel("Frequency, Hz")

    axes[1].plot(time_axis_s, prediction, label="prediction")
    axes[1].axhline(peak_threshold, color="red", linestyle="--", label="peak_threshold")
    for event_time in predicted_times_inside:
        axes[1].axvline(event_time, color="tab:blue", linewidth=1.0, alpha=0.5)
    axes[1].set_xlabel("Time inside segment, s")
    axes[1].set_ylabel("Predicted heatmap")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "files": len(set(str(row["source_file"]) for row in rows)),
        "segments": len(rows),
        "total_predicted_events": int(sum(int(row["predicted_event_count"]) for row in rows)),
        "by_channel": {},
        "by_file_channel": {},
    }
    for channel in sorted(set(str(row["channel"]) for row in rows)):
        channel_rows = [row for row in rows if row["channel"] == channel]
        counts = np.asarray([int(row["predicted_event_count"]) for row in channel_rows], dtype=np.float32)
        summary["by_channel"][channel] = {
            "segments": int(len(channel_rows)),
            "total_predicted_events": int(counts.sum()),
            "mean_predicted_events": float(counts.mean()) if counts.size else 0.0,
            "max_predicted_events": int(counts.max()) if counts.size else 0,
        }
    for key in sorted(set((str(row["source_file"]), str(row["channel"])) for row in rows)):
        file_path, channel = key
        key_rows = [row for row in rows if str(row["source_file"]) == file_path and str(row["channel"]) == channel]
        counts = np.asarray([int(row["predicted_event_count"]) for row in key_rows], dtype=np.float32)
        summary["by_file_channel"][f"{Path(file_path).name}:{channel}"] = {
            "segments": int(len(key_rows)),
            "total_predicted_events": int(counts.sum()),
            "mean_predicted_events": float(counts.mean()) if counts.size else 0.0,
            "max_predicted_events": int(counts.max()) if counts.size else 0,
        }
    return summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    input_files = resolve_input_files(args.input, args.pattern, args.recursive)

    loc = import_module_from_path("train_sferics_localization_2d", METHOD_DIR / "train_sferics_localization_2d.py")
    broadband = import_module_from_path("plot_broadband_spectrogram", PROJECT_ROOT / "plot_broadband_spectrogram.py")
    torch_mod, _, _, _, Model = loc.build_torch_components()
    device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    checkpoint = torch_mod.load(model_path, map_location=device)
    model = Model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    should_plot = args.save_plots or args.show
    plt = configure_matplotlib(show=args.show) if should_plot else None
    channels = selected_channels(args.channel)
    rows: list[dict[str, Any]] = []
    plot_items: list[dict[str, Any]] = []

    for file_path in input_files:
        record_size, sample_count, data_offset = broadband.read_metadata(file_path)
        samples = broadband.load_samples(file_path, sample_count, record_size, data_offset)
        for channel in channels:
            channel_index = broadband.CHANNEL_NAME_TO_INDEX[channel]
            channel_samples = broadband.extract_channel(
                samples=samples,
                sample_count=sample_count,
                record_size=record_size,
                channels=args.channels,
                channel_index=channel_index,
            )
            nperseg, hop = broadband.resolve_stft_parameters(
                samples=channel_samples,
                sample_rate=args.sample_rate,
                nperseg=args.nperseg,
                max_frames=args.max_frames,
                time_resolution=args.time_resolution,
                frequency_resolution=args.frequency_resolution,
            )
            freq_min, freq_max = broadband.resolve_frequency_band(args.sample_rate, args.freq_min, args.freq_max)
            segments = broadband.iter_time_segments(
                channel_samples,
                sample_rate=args.sample_rate,
                segment_duration=args.segment_duration,
                nperseg=nperseg,
            )

            for segment_index, start, stop, start_s, stop_s in segments:
                segment_samples = channel_samples[start:stop]
                spec_db, time_axis_samples, freq_axis = broadband.compute_spectrogram(
                    segment_samples,
                    nperseg=nperseg,
                    hop=hop,
                )
                spec_db, freq_axis = broadband.crop_frequency_band(
                    spec_db=spec_db,
                    freq_axis=freq_axis,
                    sample_rate=args.sample_rate,
                    freq_min=freq_min,
                    freq_max=freq_max,
                )
                time_axis_s = time_axis_samples / args.sample_rate
                freq_axis_hz = freq_axis * args.sample_rate
                prediction = predict_segment(model, torch_mod, device, loc, spec_db)
                predicted_times_inside = extract_predicted_times(
                    loc,
                    prediction,
                    time_axis_s,
                    args.peak_threshold,
                    args.min_peak_distance,
                )
                predicted_times_absolute = [float(start_s + value) for value in predicted_times_inside]
                max_heatmap = float(np.max(prediction)) if prediction.size else 0.0
                mean_heatmap = float(np.mean(prediction)) if prediction.size else 0.0

                row = {
                    "source_file": str(file_path),
                    "channel": channel,
                    "segment_index": int(segment_index),
                    "segment_start_s": float(start_s),
                    "segment_end_s": float(stop_s),
                    "peak_threshold": float(args.peak_threshold),
                    "max_heatmap": max_heatmap,
                    "mean_heatmap": mean_heatmap,
                    "predicted_event_count": int(len(predicted_times_inside)),
                    "predicted_event_times_inside_s_json": json.dumps(predicted_times_inside),
                    "predicted_event_times_absolute_s_json": json.dumps(predicted_times_absolute),
                }
                rows.append(row)

                if should_plot:
                    plot_items.append(
                        {
                            "row": row,
                            "source_file": file_path,
                            "spec_db": spec_db.astype(np.float32),
                            "time_axis_s": time_axis_s.astype(np.float32),
                            "freq_axis_hz": freq_axis_hz.astype(np.float32),
                            "prediction": prediction.astype(np.float32),
                            "predicted_times_inside": predicted_times_inside,
                        }
                    )

    predictions_path = output_dir / "sferics_localization_2d_raw_predictions.csv"
    summary_path = output_dir / "sferics_localization_2d_raw_summary.json"
    write_csv(predictions_path, rows)
    summary = {
        "model_path": str(model_path),
        "device": str(device),
        "input_files": [str(path) for path in input_files],
        "channels": channels,
        "segment_duration": args.segment_duration,
        "freq_min": args.freq_min,
        "freq_max": args.freq_max,
        "time_resolution": args.time_resolution,
        "frequency_resolution": args.frequency_resolution,
        "peak_threshold": args.peak_threshold,
        "min_peak_distance": args.min_peak_distance,
        **summarize_rows(rows),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if should_plot and plt is not None:
        if args.plot_all:
            selected_plot_items = plot_items
        else:
            grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for item in plot_items:
                row = item["row"]
                grouped.setdefault((str(row["source_file"]), str(row["channel"])), []).append(item)
            selected_plot_items = []
            for items in grouped.values():
                selected_plot_items.extend(
                    sorted(
                        items,
                        key=lambda item: (
                            int(item["row"]["predicted_event_count"]),
                            float(item["row"]["max_heatmap"]),
                        ),
                        reverse=True,
                    )[: max(0, args.plot_top)]
                )

        for item in selected_plot_items:
            row = item["row"]
            plot_output = None
            if args.save_plots:
                plot_output = (
                    output_dir
                    / "plots"
                    / output_stem(Path(str(row["source_file"])))
                    / str(row["channel"])
                    / f"seg{int(row['segment_index']):03d}_{float(row['segment_start_s']):07.3f}-{float(row['segment_end_s']):07.3f}s.png"
                )
            render_prediction_plot(
                plt,
                output=plot_output,
                show=args.show,
                source_file=Path(str(row["source_file"])),
                channel=str(row["channel"]),
                segment_index=int(row["segment_index"]),
                segment_start_s=float(row["segment_start_s"]),
                segment_end_s=float(row["segment_end_s"]),
                spec_db=item["spec_db"],
                time_axis_s=item["time_axis_s"],
                freq_axis_hz=item["freq_axis_hz"],
                prediction=item["prediction"],
                predicted_times_inside=item["predicted_times_inside"],
                peak_threshold=float(args.peak_threshold),
                db_low=args.db_low,
                db_high=args.db_high,
                cmap=args.cmap,
                dpi=args.dpi,
            )

    print(f"Model         : {model_path}")
    print(f"Device        : {device}")
    print(f"Input files   : {len(input_files)}")
    print(f"Segments      : {len(rows)}")
    print(f"Predictions   : {predictions_path}")
    print(f"Summary       : {summary_path}")
    if args.save_plots:
        print(f"Plots         : {output_dir / 'plots'}")
    print("By channel:")
    for channel, data in summary["by_channel"].items():
        print(
            f"  {channel}: segments={data['segments']} "
            f"events={data['total_predicted_events']} "
            f"mean={data['mean_predicted_events']:.2f} "
            f"max={data['max_predicted_events']}"
        )


if __name__ == "__main__":
    main()
