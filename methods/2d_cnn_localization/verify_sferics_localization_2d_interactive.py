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
DEFAULT_OUTPUT_DIR = METHOD_DIR / "reports" / "raw_verification"
DEFAULT_PATTERN = "Broadband_Data_*.bin"
STATUS_VALUES = {
    "model_verified",
    "manual_corrected",
    "false_positive",
    "missed_events",
    "artifact_suspected",
    "uncertain",
}


FIELDNAMES = [
    "verification_id",
    "source_file",
    "channel",
    "segment_index",
    "segment_start_s",
    "segment_end_s",
    "peak_threshold",
    "max_heatmap",
    "mean_heatmap",
    "predicted_event_count",
    "predicted_event_times_inside_s_json",
    "predicted_event_times_absolute_s_json",
    "review_status",
    "corrected_event_count",
    "corrected_event_times_inside_s_json",
    "corrected_event_times_absolute_s_json",
    "quality_flag",
    "comment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively verify the saved 2D sferics localization model on raw "
            "Broadband_Data files. The model predicts event times; the user confirms "
            "or corrects them in a matplotlib window. Results are saved to CSV."
        )
    )
    parser.add_argument("input", type=Path, help="Broadband_Data_*.bin file or directory.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help=f"Model checkpoint. Default: {DEFAULT_MODEL}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Output dir. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--verification-csv", type=Path, default=None, help="CSV path. Default: <output-dir>/verification.csv")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help=f"Pattern for directory input. Default: {DEFAULT_PATTERN}")
    parser.add_argument("--recursive", action="store_true", help="Search directory recursively.")
    parser.add_argument("--channel", choices=("ns", "we", "both"), default="both", help="Channel to verify. Default: both")
    parser.add_argument("--sample-rate", type=float, default=100_000.0, help="Per-channel sample rate in Hz.")
    parser.add_argument("--channels", type=int, default=2, help="Number of interleaved channels. Default: 2")
    parser.add_argument("--segment-duration", type=float, default=2.0, help="Segment duration in seconds.")
    parser.add_argument("--time-resolution", type=float, default=0.002, help="Spectrogram time step in seconds.")
    parser.add_argument("--frequency-resolution", type=float, default=50.0, help="Spectrogram frequency step in Hz.")
    parser.add_argument("--freq-min", type=float, default=20_000.0, help="Lower frequency bound in Hz.")
    parser.add_argument("--freq-max", type=float, default=30_000.0, help="Upper frequency bound in Hz.")
    parser.add_argument("--peak-threshold", type=float, default=0.35, help="Peak threshold for model event extraction.")
    parser.add_argument("--min-peak-distance", type=float, default=0.03, help="Minimum predicted peak distance in seconds.")
    parser.add_argument("--nperseg", type=int, default=4096, help="Fallback FFT window length.")
    parser.add_argument("--max-frames", type=int, default=4000, help="Maximum frames for automatic hop fallback.")
    parser.add_argument("--db-low", type=float, default=2.0, help="Lower percentile for spectrogram display.")
    parser.add_argument("--db-high", type=float, default=99.8, help="Upper percentile for spectrogram display.")
    parser.add_argument("--cmap", default="jet", help="Spectrogram colormap. Default: jet")
    parser.add_argument(
        "--include-reviewed",
        action="store_true",
        help="Include rows already having review_status. Default: only pending rows.",
    )
    parser.add_argument(
        "--review-status",
        default=None,
        help="Verify only rows with this status. Use 'pending' for blank status.",
    )
    parser.add_argument("--start-id", default=None, help="Start from this verification_id.")
    return parser.parse_args()


def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
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


def selected_channels(channel: str) -> list[str]:
    return ["ns", "we"] if channel == "both" else [channel]


def verification_id(file_path: Path, channel: str, segment_index: int) -> str:
    return f"{file_path.stem}_{channel}_seg{segment_index:03d}"


def parse_times(raw: str | None) -> list[float]:
    text = (raw or "").strip()
    if not text:
        return []
    values = json.loads(text)
    if not isinstance(values, list):
        raise ValueError("Expected JSON list of event times.")
    return sorted(float(value) for value in values)


def format_times(times: list[float]) -> str:
    return json.dumps([round(float(value), 6) for value in sorted(times)])


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def predict_segment(model, torch_mod, device, loc, spec_db: np.ndarray) -> np.ndarray:
    normalized = loc.normalize_spec(spec_db)
    with torch_mod.no_grad():
        prediction = model(torch_mod.from_numpy(normalized[None, None, :, :]).to(device))
    return prediction.detach().cpu().numpy()[0].astype(np.float32)


def pick_times(loc, prediction: np.ndarray, time_axis_s: np.ndarray, threshold: float, min_distance_s: float) -> list[float]:
    time_step = float(np.median(np.diff(time_axis_s))) if time_axis_s.size > 1 else 0.002
    min_distance_bins = max(1, int(round(min_distance_s / time_step)))
    indices = loc.pick_peaks(prediction, threshold=threshold, min_distance_bins=min_distance_bins)
    return [float(time_axis_s[index]) for index in indices]


def build_or_update_rows(
    *,
    files: list[Path],
    channels: list[str],
    verification_csv: Path,
    args: argparse.Namespace,
    broadband,
    loc,
    torch_mod,
    model,
    device,
) -> list[dict[str, Any]]:
    existing_rows = {row["verification_id"]: row for row in read_csv_rows(verification_csv)}
    rows: list[dict[str, Any]] = []

    for file_path in files:
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
            segments = broadband.iter_time_segments(channel_samples, args.sample_rate, args.segment_duration, nperseg)
            freq_min, freq_max = broadband.resolve_frequency_band(args.sample_rate, args.freq_min, args.freq_max)

            for segment_index, start, stop, start_s, stop_s in segments:
                row_id = verification_id(file_path, channel, segment_index)
                if row_id in existing_rows:
                    rows.append(existing_rows[row_id])
                    continue
                spec_db, time_axis_samples, freq_axis = broadband.compute_spectrogram(
                    channel_samples[start:stop],
                    nperseg=nperseg,
                    hop=hop,
                )
                spec_db, _ = broadband.crop_frequency_band(spec_db, freq_axis, args.sample_rate, freq_min, freq_max)
                time_axis_s = time_axis_samples / args.sample_rate
                prediction = predict_segment(model, torch_mod, device, loc, spec_db)
                predicted_inside = pick_times(
                    loc,
                    prediction,
                    time_axis_s,
                    args.peak_threshold,
                    args.min_peak_distance,
                )
                predicted_absolute = [float(start_s + value) for value in predicted_inside]
                rows.append(
                    {
                        "verification_id": row_id,
                        "source_file": str(file_path),
                        "channel": channel,
                        "segment_index": int(segment_index),
                        "segment_start_s": float(start_s),
                        "segment_end_s": float(stop_s),
                        "peak_threshold": float(args.peak_threshold),
                        "max_heatmap": float(np.max(prediction)) if prediction.size else 0.0,
                        "mean_heatmap": float(np.mean(prediction)) if prediction.size else 0.0,
                        "predicted_event_count": len(predicted_inside),
                        "predicted_event_times_inside_s_json": format_times(predicted_inside),
                        "predicted_event_times_absolute_s_json": format_times(predicted_absolute),
                        "review_status": "",
                        "corrected_event_count": "",
                        "corrected_event_times_inside_s_json": "",
                        "corrected_event_times_absolute_s_json": "",
                        "quality_flag": "",
                        "comment": "",
                    }
                )
    write_csv_rows(verification_csv, rows)
    return rows


class RawVerificationReviewer:
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        order: list[int],
        verification_csv: Path,
        args: argparse.Namespace,
        broadband,
        loc,
        torch_mod,
        model,
        device,
    ) -> None:
        self.rows = rows
        self.order = order
        self.verification_csv = verification_csv
        self.args = args
        self.broadband = broadband
        self.loc = loc
        self.torch_mod = torch_mod
        self.model = model
        self.device = device
        self.position = 0
        self.clicked_times: list[float] = []
        self.predicted_times: list[float] = []
        self.time_min = 0.0
        self.time_max = 0.0
        self.spec_db: np.ndarray | None = None
        self.time_axis_s: np.ndarray | None = None
        self.freq_axis_hz: np.ndarray | None = None
        self.prediction: np.ndarray | None = None
        self.fig = None
        self.ax_spec = None
        self.ax_heatmap = None
        self._file_cache: dict[Path, tuple[int, int, int, np.ndarray]] = {}

    def run(self) -> None:
        plt = configure_matplotlib()
        if not self.order:
            print("No matching segments to verify.")
            return
        self.fig, (self.ax_spec, self.ax_heatmap) = plt.subplots(
            2,
            1,
            figsize=(12, 7),
            sharex=True,
            constrained_layout=True,
        )
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.show_current()
        plt.show()

    def current_row(self) -> dict[str, Any]:
        return self.rows[self.order[self.position]]

    def load_file_samples(self, file_path: Path) -> tuple[int, int, int, np.ndarray]:
        if file_path not in self._file_cache:
            record_size, sample_count, data_offset = self.broadband.read_metadata(file_path)
            samples = self.broadband.load_samples(file_path, sample_count, record_size, data_offset)
            self._file_cache[file_path] = (record_size, sample_count, data_offset, samples)
        return self._file_cache[file_path]

    def load_current_segment(self) -> None:
        row = self.current_row()
        file_path = Path(str(row["source_file"]))
        channel = str(row["channel"])
        segment_start_s = float(row["segment_start_s"])
        record_size, sample_count, _, samples = self.load_file_samples(file_path)
        channel_samples = self.broadband.extract_channel(
            samples=samples,
            sample_count=sample_count,
            record_size=record_size,
            channels=self.args.channels,
            channel_index=self.broadband.CHANNEL_NAME_TO_INDEX[channel],
        )
        nperseg, hop = self.broadband.resolve_stft_parameters(
            samples=channel_samples,
            sample_rate=self.args.sample_rate,
            nperseg=self.args.nperseg,
            max_frames=self.args.max_frames,
            time_resolution=self.args.time_resolution,
            frequency_resolution=self.args.frequency_resolution,
        )
        start = int(round(segment_start_s * self.args.sample_rate))
        stop = start + int(round(self.args.segment_duration * self.args.sample_rate))
        freq_min, freq_max = self.broadband.resolve_frequency_band(self.args.sample_rate, self.args.freq_min, self.args.freq_max)
        spec_db, time_axis_samples, freq_axis = self.broadband.compute_spectrogram(channel_samples[start:stop], nperseg, hop)
        spec_db, freq_axis = self.broadband.crop_frequency_band(spec_db, freq_axis, self.args.sample_rate, freq_min, freq_max)
        self.spec_db = spec_db.astype(np.float32)
        self.time_axis_s = (time_axis_samples / self.args.sample_rate).astype(np.float32)
        self.freq_axis_hz = (freq_axis * self.args.sample_rate).astype(np.float32)
        self.prediction = predict_segment(self.model, self.torch_mod, self.device, self.loc, self.spec_db)
        self.predicted_times = parse_times(str(row["predicted_event_times_inside_s_json"]))
        if str(row.get("corrected_event_times_inside_s_json", "")).strip():
            self.clicked_times = parse_times(str(row["corrected_event_times_inside_s_json"]))
        else:
            self.clicked_times = list(self.predicted_times)
        self.time_min = float(self.time_axis_s[0])
        self.time_max = float(self.time_axis_s[-1])

    def redraw(self) -> None:
        if self.ax_spec is None or self.ax_heatmap is None:
            raise RuntimeError("Figure axes are not initialized.")
        if self.spec_db is None or self.time_axis_s is None or self.freq_axis_hz is None or self.prediction is None:
            raise RuntimeError("Current segment is not loaded.")
        row = self.current_row()
        vmin, vmax = np.percentile(self.spec_db, [self.args.db_low, self.args.db_high])
        self.ax_spec.clear()
        self.ax_heatmap.clear()
        self.ax_spec.imshow(
            self.spec_db,
            origin="lower",
            aspect="auto",
            cmap=self.args.cmap,
            vmin=vmin,
            vmax=vmax,
            extent=(self.time_min, self.time_max, float(self.freq_axis_hz[0]), float(self.freq_axis_hz[-1])),
            interpolation="nearest",
        )
        for event_time in self.predicted_times:
            self.ax_spec.axvline(event_time, color="white", linestyle="--", linewidth=1.0, alpha=0.9)
        for event_time in self.clicked_times:
            self.ax_spec.axvline(event_time, color="red", linestyle="-", linewidth=1.2, alpha=0.95)

        self.ax_heatmap.plot(self.time_axis_s, self.prediction, label="prediction")
        self.ax_heatmap.axhline(float(row["peak_threshold"]), color="red", linestyle="--", label="peak_threshold")
        for event_time in self.predicted_times:
            self.ax_heatmap.axvline(event_time, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        for event_time in self.clicked_times:
            self.ax_heatmap.axvline(event_time, color="tab:red", linewidth=1.0, alpha=0.7)
        self.ax_heatmap.set_xlabel("Time inside segment, s")
        self.ax_heatmap.set_ylabel("Predicted heatmap")
        self.ax_heatmap.grid(alpha=0.3)
        self.ax_heatmap.legend(loc="upper right")

        status = str(row.get("review_status", "")).strip() or "pending"
        self.ax_spec.set_title(
            f"{self.position + 1}/{len(self.order)} {row['verification_id']} | "
            f"{row['channel']} | predicted={row['predicted_event_count']} | status={status}\n"
            "white=model, red=your correction. Keys: v=verified, enter/c=corrected, "
            "f=false positive, m=missed events, a=artifact, u=uncertain, "
            "backspace=undo, r=reset model, x=clear, s=skip, q=quit",
            fontsize=10,
        )
        self.ax_spec.set_ylabel("Frequency, Hz")
        self.ax_spec.figure.canvas.draw_idle()
        print(
            f"[{self.position + 1}/{len(self.order)}] {row['verification_id']} "
            f"predicted={self.predicted_times} selected={self.clicked_times}"
        )

    def show_current(self) -> None:
        self.load_current_segment()
        self.redraw()

    def on_click(self, event) -> None:
        if event.xdata is None or event.inaxes not in {self.ax_spec, self.ax_heatmap}:
            return
        event_time = min(max(float(event.xdata), self.time_min), self.time_max)
        self.clicked_times.append(event_time)
        self.clicked_times.sort()
        self.redraw()

    def on_key(self, event) -> None:
        key = event.key or ""
        if key == "q":
            print("Stopping verification.")
            if self.fig is not None:
                import matplotlib.pyplot as plt

                plt.close(self.fig)
            return
        if key == "s":
            self.advance()
            return
        if key in ("backspace", "delete"):
            if self.clicked_times:
                removed = self.clicked_times.pop()
                print(f"Removed time {removed:.6f}")
            self.redraw()
            return
        if key == "r":
            self.clicked_times = list(self.predicted_times)
            self.redraw()
            return
        if key == "x":
            self.clicked_times = []
            self.redraw()
            return
        if key == "v":
            self.save_current("model_verified", corrected_times=self.predicted_times, quality_flag="")
            self.advance()
            return
        if key in ("enter", "c"):
            self.save_current("manual_corrected", corrected_times=self.clicked_times, quality_flag="")
            self.advance()
            return
        if key == "f":
            self.save_current("false_positive", corrected_times=[], quality_flag="")
            self.advance()
            return
        if key == "m":
            self.save_current("missed_events", corrected_times=self.clicked_times, quality_flag="")
            self.advance()
            return
        if key == "a":
            self.save_current("artifact_suspected", corrected_times=self.clicked_times, quality_flag="artifact_suspected")
            self.advance()
            return
        if key == "u":
            self.save_current("uncertain", corrected_times=self.clicked_times, quality_flag="uncertain")
            self.advance()
            return

    def save_current(self, review_status: str, corrected_times: list[float], quality_flag: str) -> None:
        row = self.current_row()
        start_s = float(row["segment_start_s"])
        corrected_absolute = [start_s + value for value in corrected_times]
        row["review_status"] = review_status
        row["corrected_event_count"] = len(corrected_times)
        row["corrected_event_times_inside_s_json"] = format_times(corrected_times)
        row["corrected_event_times_absolute_s_json"] = format_times(corrected_absolute)
        row["quality_flag"] = quality_flag
        write_csv_rows(self.verification_csv, self.rows)
        print(f"Saved {row['verification_id']}: {review_status}")

    def advance(self) -> None:
        self.position += 1
        if self.position >= len(self.order):
            print("Verification queue complete.")
            if self.fig is not None:
                import matplotlib.pyplot as plt

                plt.close(self.fig)
            return
        self.show_current()


def build_order(rows: list[dict[str, Any]], include_reviewed: bool, review_status: str | None) -> list[int]:
    order: list[int] = []
    for index, row in enumerate(rows):
        current_status = str(row.get("review_status", "")).strip() or "pending"
        if review_status is not None and current_status != review_status:
            continue
        if review_status is None and not include_reviewed and current_status != "pending":
            continue
        order.append(index)
    return order


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    verification_csv = (
        args.verification_csv.expanduser().resolve()
        if args.verification_csv
        else output_dir / "sferics_localization_2d_raw_verification.csv"
    )
    files = resolve_input_files(args.input, args.pattern, args.recursive)
    channels = selected_channels(args.channel)

    loc = import_module_from_path("train_sferics_localization_2d", METHOD_DIR / "train_sferics_localization_2d.py")
    broadband = import_module_from_path("plot_broadband_spectrogram", PROJECT_ROOT / "plot_broadband_spectrogram.py")
    torch_mod, _, _, _, Model = loc.build_torch_components()
    device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    checkpoint = torch_mod.load(args.model.expanduser().resolve(), map_location=device)
    model = Model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = build_or_update_rows(
        files=files,
        channels=channels,
        verification_csv=verification_csv,
        args=args,
        broadband=broadband,
        loc=loc,
        torch_mod=torch_mod,
        model=model,
        device=device,
    )
    order = build_order(rows, args.include_reviewed or (args.review_status not in (None, "pending")), args.review_status)
    if args.start_id:
        positions = [pos for pos, row_index in enumerate(order) if rows[row_index]["verification_id"] == args.start_id]
        if not positions:
            raise ValueError(f"start-id is not in the current queue: {args.start_id}")
        order = order[positions[0] :]

    print(f"Verification CSV: {verification_csv}")
    print(f"Total rows      : {len(rows)}")
    print(f"Queue rows      : {len(order)}")
    print(f"Device          : {device}")
    reviewer = RawVerificationReviewer(
        rows=rows,
        order=order,
        verification_csv=verification_csv,
        args=args,
        broadband=broadband,
        loc=loc,
        torch_mod=torch_mod,
        model=model,
        device=device,
    )
    reviewer.run()


if __name__ == "__main__":
    main()
