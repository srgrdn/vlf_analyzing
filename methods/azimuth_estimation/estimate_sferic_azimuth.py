#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


METHOD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = METHOD_DIR.parent.parent
DEFAULT_EVENTS_CSV = PROJECT_ROOT / "full_result" / "predictions" / "full_raw_2d_cnn_predictions.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "azimuth_results"
DEFAULT_SAMPLE_RATE = 100_000.0
DEFAULT_CHANNELS = 2
CHANNEL_INDEX = {"ns": 0, "we": 1}
FILENAME_RE = re.compile(r"Broadband_Data_(\d{4})\.(\d{2})\.(\d{2})_(\d{2})\.(\d{2})\.(\d{2})")


@dataclass
class EventDetection:
    source_file: Path
    event_time_s: float
    channel: str


@dataclass
class MergedEvent:
    source_file: Path
    event_time_s: float
    channels: tuple[str, ...]
    detections: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental single-station azimuth/line-of-bearing estimation for sferics "
            "using two orthogonal loop antenna channels (`ns`, `we`). This estimates only "
            "an arrival axis with 180-degree ambiguity, not source coordinates or range."
        )
    )
    parser.add_argument(
        "--events-csv",
        type=Path,
        default=DEFAULT_EVENTS_CSV,
        help=f"Prediction/event CSV with event times. Default: {DEFAULT_EVENTS_CSV}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE, help="Per-channel sample rate in Hz.")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS, help="Number of interleaved channels.")
    parser.add_argument(
        "--merge-tolerance",
        type=float,
        default=0.04,
        help="Merge ns/we detections into one physical event if times differ by this many seconds. Default: 0.04",
    )
    parser.add_argument(
        "--analysis-window-ms",
        type=float,
        default=5.0,
        help="Central event window half-width for PCA/RMS analysis, in milliseconds. Default: 5",
    )
    parser.add_argument(
        "--filter-window-ms",
        type=float,
        default=50.0,
        help="Larger half-window used before filtering, in milliseconds. Default: 50",
    )
    parser.add_argument("--freq-min", type=float, default=20_000.0, help="Band-pass lower edge in Hz. Default: 20000")
    parser.add_argument("--freq-max", type=float, default=30_000.0, help="Band-pass upper edge in Hz. Default: 30000")
    parser.add_argument("--no-filter", action="store_true", help="Disable band-pass filtering.")
    parser.add_argument("--gain-ns", type=float, default=1.0, help="Amplitude calibration multiplier for ns.")
    parser.add_argument("--gain-we", type=float, default=1.0, help="Amplitude calibration multiplier for we.")
    parser.add_argument(
        "--phase-shift-samples",
        type=int,
        default=0,
        help=(
            "Integer sample shift applied to `we` relative to `ns`. Positive values delay `we`; "
            "negative values advance it. Default: 0"
        ),
    )
    parser.add_argument(
        "--antenna-angle-offset-deg",
        type=float,
        default=0.0,
        help="Constant rotation added to the arrival axis, in degrees. Default: 0",
    )
    parser.add_argument("--min-energy", type=float, default=1e-4, help="Low-energy quality threshold after filtering.")
    parser.add_argument("--min-snr", type=float, default=3.0, help="Low-SNR quality threshold.")
    parser.add_argument("--min-linearity", type=float, default=3.0, help="Weak-PCA-linearity quality threshold.")
    parser.add_argument(
        "--saturation-level",
        type=float,
        default=32000.0,
        help="Absolute raw ADC level treated as saturation. Default: 32000",
    )
    parser.add_argument("--max-events", type=int, default=None, help="Optional cap for debugging/smoke runs.")
    parser.add_argument("--example-count", type=int, default=12, help="Number of diagnostic good-event examples to plot.")
    parser.add_argument("--plot-dpi", type=int, default=160, help="Output plot DPI.")
    parser.add_argument("--receiver-lat", type=float, default=52.2864, help="Receiver latitude. Default: Irkutsk.")
    parser.add_argument("--receiver-lon", type=float, default=104.2807, help="Receiver longitude. Default: Irkutsk.")
    parser.add_argument("--map-radius-km", type=float, default=650.0, help="Radius for the bearing-line map in km.")
    return parser.parse_args()


def import_project_reader():
    sys.path.insert(0, str(PROJECT_ROOT))
    import plot_broadband_spectrogram as broadband

    return broadband


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def import_scipy_signal():
    try:
        from scipy import signal
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("scipy is required for optional band-pass filtering.") from exc
    return signal


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Events CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_json_list(raw: str | None) -> list[float]:
    if not raw:
        return []
    values = json.loads(raw)
    if not isinstance(values, list):
        raise ValueError(f"Expected JSON list, got: {raw!r}")
    return [float(value) for value in values]


def load_event_detections(events_csv: Path, max_events: int | None) -> list[EventDetection]:
    rows = read_csv_rows(events_csv)
    detections: list[EventDetection] = []
    for row in rows:
        if row.get("error"):
            continue
        source = row.get("source_file")
        channel = row.get("channel")
        if not source or channel not in CHANNEL_INDEX:
            continue

        absolute_raw = row.get("predicted_event_times_absolute_s_json")
        inside_raw = row.get("predicted_event_times_inside_s_json")
        if absolute_raw:
            times = parse_json_list(absolute_raw)
        else:
            segment_start = float(row.get("segment_start_s", 0.0))
            times = [segment_start + value for value in parse_json_list(inside_raw)]

        for event_time in times:
            detections.append(EventDetection(Path(source), float(event_time), str(channel)))
            if max_events is not None and len(detections) >= max_events:
                return detections
    return detections


def merge_detections(detections: list[EventDetection], tolerance_s: float) -> list[MergedEvent]:
    grouped: dict[Path, list[EventDetection]] = defaultdict(list)
    for detection in detections:
        grouped[detection.source_file].append(detection)

    merged: list[MergedEvent] = []
    for source_file, file_detections in sorted(grouped.items(), key=lambda item: str(item[0])):
        ordered = sorted(file_detections, key=lambda detection: detection.event_time_s)
        current: list[EventDetection] = []
        for detection in ordered:
            if not current:
                current = [detection]
                continue
            if detection.event_time_s - current[-1].event_time_s <= tolerance_s:
                current.append(detection)
            else:
                merged.append(make_merged_event(source_file, current))
                current = [detection]
        if current:
            merged.append(make_merged_event(source_file, current))
    return merged


def make_merged_event(source_file: Path, detections: list[EventDetection]) -> MergedEvent:
    return MergedEvent(
        source_file=source_file,
        event_time_s=float(np.mean([detection.event_time_s for detection in detections])),
        channels=tuple(sorted(set(detection.channel for detection in detections))),
        detections=len(detections),
    )


def parse_datetime_from_filename(path: Path) -> datetime | None:
    match = FILENAME_RE.search(path.name)
    if not match:
        return None
    year, month, day, hour, minute, second = map(int, match.groups())
    return datetime(year, month, day, hour, minute, second)


def shift_signal(values: np.ndarray, shift_samples: int) -> np.ndarray:
    if shift_samples == 0:
        return values
    shifted = np.zeros_like(values)
    if shift_samples > 0:
        shifted[shift_samples:] = values[:-shift_samples]
    else:
        shifted[:shift_samples] = values[-shift_samples:]
    return shifted


def safe_slice(values: np.ndarray, start: int, stop: int) -> np.ndarray:
    start = max(0, int(start))
    stop = min(int(stop), values.size)
    if stop <= start:
        return np.array([], dtype=np.float32)
    return np.asarray(values[start:stop], dtype=np.float32)


def compute_snr(signal_window: np.ndarray, noise_window: np.ndarray, eps: float = 1e-12) -> float:
    signal_power = float(np.mean(signal_window**2)) if signal_window.size else 0.0
    noise_power = float(np.mean(noise_window**2)) if noise_window.size else 0.0
    return signal_power / (noise_power + eps)


def estimate_event(
    event: MergedEvent,
    ns_samples: np.ndarray,
    we_samples: np.ndarray,
    *,
    sample_rate: float,
    analysis_window_s: float,
    filter_window_s: float,
    bandpass_sos,
    signal_mod,
    gain_ns: float,
    gain_we: float,
    phase_shift_samples: int,
    antenna_angle_offset_deg: float,
    min_energy: float,
    min_snr: float,
    min_linearity: float,
    saturation_level: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    eps = 1e-12
    center = int(round(event.event_time_s * sample_rate))
    filter_half = max(int(round(filter_window_s * sample_rate)), int(round(analysis_window_s * sample_rate)))
    analysis_half = int(round(analysis_window_s * sample_rate))

    raw_start = center - filter_half
    raw_stop = center + filter_half + 1
    raw_ns = safe_slice(ns_samples, raw_start, raw_stop)
    raw_we = safe_slice(we_samples, raw_start, raw_stop)
    n = min(raw_ns.size, raw_we.size)
    raw_ns = raw_ns[:n]
    raw_we = raw_we[:n]
    if n < max(8, 2 * analysis_half + 1):
        raise ValueError("Event window is too close to file boundary or too short.")

    is_saturated = bool(np.any(np.abs(raw_ns) >= saturation_level) or np.any(np.abs(raw_we) >= saturation_level))

    raw_ns = raw_ns * float(gain_ns)
    raw_we = shift_signal(raw_we * float(gain_we), phase_shift_samples)

    raw_ns = raw_ns - float(np.mean(raw_ns))
    raw_we = raw_we - float(np.mean(raw_we))

    if bandpass_sos is not None:
        ns_filtered = signal_mod.sosfiltfilt(bandpass_sos, raw_ns).astype(np.float32)
        we_filtered = signal_mod.sosfiltfilt(bandpass_sos, raw_we).astype(np.float32)
    else:
        ns_filtered = raw_ns.astype(np.float32)
        we_filtered = raw_we.astype(np.float32)

    local_center = min(max(center - max(0, raw_start), analysis_half), n - analysis_half - 1)
    central_slice = slice(local_center - analysis_half, local_center + analysis_half + 1)
    ns = ns_filtered[central_slice]
    we = we_filtered[central_slice]

    noise_mask = np.ones(n, dtype=bool)
    noise_mask[central_slice] = False
    noise_ns = ns_filtered[noise_mask]
    noise_we = we_filtered[noise_mask]

    ns = ns - float(np.mean(ns))
    we = we - float(np.mean(we))

    a_ns = float(np.sqrt(np.mean(ns**2))) if ns.size else 0.0
    a_we = float(np.sqrt(np.mean(we**2))) if we.size else 0.0
    ratio = a_we / (a_ns + eps)
    energy_ns = float(np.sum(ns**2))
    energy_we = float(np.sum(we**2))
    snr_ns = compute_snr(ns, noise_ns, eps)
    snr_we = compute_snr(we, noise_we, eps)

    # The PCA axis describes the dominant magnetic-field polarization axis in the
    # two loop-antenna channels. This is not a source coordinate. With one station
    # and two orthogonal loops it gives only a line of possible arrival direction
    # with a 180-degree ambiguity.
    x = np.column_stack([ns, we]).astype(np.float64)
    x -= x.mean(axis=0, keepdims=True)
    covariance = np.cov(x, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    principal = eigvecs[:, 0]
    magnetic_axis_deg = math.degrees(math.atan2(float(principal[1]), float(principal[0]))) % 180.0
    arrival_axis_deg = (magnetic_axis_deg + 90.0 + antenna_angle_offset_deg) % 180.0
    azimuth_candidate_1 = arrival_axis_deg
    azimuth_candidate_2 = (arrival_axis_deg + 180.0) % 360.0
    lambda1 = float(eigvals[0])
    lambda2 = float(eigvals[1]) if eigvals.size > 1 else 0.0
    linearity = lambda1 / (lambda2 + eps)

    flags: list[str] = []
    if energy_ns < min_energy and energy_we < min_energy:
        flags.append("low_energy")
    if snr_ns < min_snr and snr_we < min_snr:
        flags.append("low_snr")
    if linearity < min_linearity:
        flags.append("weak_linearity")
    if is_saturated:
        flags.append("saturated")
    if not set(event.channels) >= {"ns", "we"}:
        flags.append("single_channel_detection")
    quality_flag = "good" if not flags else ";".join(flags)

    file_dt = parse_datetime_from_filename(event.source_file)
    event_dt = file_dt + timedelta(seconds=event.event_time_s) if file_dt else None
    result = {
        "source_file": str(event.source_file),
        "event_time_s": event.event_time_s,
        "event_datetime_utc_if_available": event_dt.isoformat() if event_dt else "",
        "detected_channels": "|".join(event.channels),
        "merged_detection_count": int(event.detections),
        "A_ns": a_ns,
        "A_we": a_we,
        "ratio_we_ns": ratio,
        "energy_ns": energy_ns,
        "energy_we": energy_we,
        "snr_ns": snr_ns,
        "snr_we": snr_we,
        "magnetic_axis_deg": magnetic_axis_deg,
        "arrival_axis_deg": arrival_axis_deg,
        "azimuth_candidate_1": azimuth_candidate_1,
        "azimuth_candidate_2": azimuth_candidate_2,
        "pca_lambda1": lambda1,
        "pca_lambda2": lambda2,
        "pca_linearity": linearity,
        "is_saturated": int(is_saturated),
        "quality_flag": quality_flag,
    }
    diagnostic = {
        "ns": ns,
        "we": we,
        "time_ms": (np.arange(ns.size) - analysis_half) / sample_rate * 1000.0,
        "principal": principal,
    }
    return result, diagnostic


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    good = [row for row in rows if row["quality_flag"] == "good"]
    angles = np.asarray([float(row["arrival_axis_deg"]) for row in good], dtype=float)
    summary: dict[str, Any] = {
        "events_processed": len(rows),
        "reliable_events": len(good),
        "quality_counts": dict(sorted(count_values(row["quality_flag"] for row in rows).items())),
    }
    if angles.size:
        hist, edges = np.histogram(angles, bins=np.arange(0, 181, 10))
        peak_index = int(np.argmax(hist))
        summary.update(
            {
                "peak_bin_start_deg": float(edges[peak_index]),
                "peak_bin_end_deg": float(edges[peak_index + 1]),
                "peak_bin_count": int(hist[peak_index]),
                "mean_arrival_axis_deg": circular_axis_mean_180(angles),
            }
        )
    return summary


def count_values(values) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(counts)


def circular_axis_mean_180(angles_deg: np.ndarray) -> float:
    doubled = np.deg2rad(2.0 * angles_deg)
    mean_angle = 0.5 * math.atan2(float(np.mean(np.sin(doubled))), float(np.mean(np.cos(doubled))))
    return math.degrees(mean_angle) % 180.0


def destination_point(lat_deg: float, lon_deg: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    radius_km = 6371.0
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    bearing = math.radians(bearing_deg)
    angular = distance_km / radius_km

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular)
        + math.cos(lat1) * math.sin(angular) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(lat1),
        math.cos(angular) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), ((math.degrees(lon2) + 540.0) % 360.0) - 180.0


def make_plots(rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    plt = configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    good = [row for row in rows if row["quality_flag"] == "good"]
    plot_rows = good if good else rows
    angles = np.asarray([float(row["arrival_axis_deg"]) for row in plot_rows], dtype=float)
    ratios = np.asarray([float(row["ratio_we_ns"]) for row in rows], dtype=float)
    linearity = np.asarray([float(row["pca_linearity"]) for row in rows], dtype=float)

    plt.figure(figsize=(9, 5))
    plt.hist(angles, bins=np.arange(0, 181, 10), color="#4c78a8", edgecolor="white")
    plt.title("Arrival axis histogram (0-180 deg)")
    plt.xlabel("Arrival axis, deg")
    plt.ylabel("Events")
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / "azimuth_histogram.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    bins = np.deg2rad(np.arange(0, 181, 10))
    hist, edges = np.histogram(np.deg2rad(angles), bins=bins)
    widths = np.diff(edges)
    ax.bar(edges[:-1], hist, width=widths, align="edge", color="#4c78a8", alpha=0.75, edgecolor="white")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Arrival axis rose (180-degree ambiguity)")
    plt.savefig(output_dir / "azimuth_rose.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(ratios[np.isfinite(ratios)], bins=60, color="#f58518", edgecolor="white")
    plt.title("A_we / A_ns histogram")
    plt.xlabel("ratio_we_ns")
    plt.ylabel("Events")
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / "ratio_we_ns_histogram.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    finite_linearity = linearity[np.isfinite(linearity)]
    finite_linearity = finite_linearity[finite_linearity < np.percentile(finite_linearity, 99.5)] if finite_linearity.size else finite_linearity
    plt.hist(finite_linearity, bins=60, color="#54a24b", edgecolor="white")
    plt.title("PCA linearity histogram")
    plt.xlabel("lambda1 / lambda2")
    plt.ylabel("Events")
    plt.grid(alpha=0.3)
    plt.savefig(output_dir / "pca_linearity_histogram.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    times = []
    y_angles = []
    for row in plot_rows:
        raw = row.get("event_datetime_utc_if_available", "")
        if raw:
            times.append(datetime.fromisoformat(str(raw)))
            y_angles.append(float(row["arrival_axis_deg"]))
    if times:
        plt.figure(figsize=(12, 5))
        plt.scatter(times, y_angles, s=8, alpha=0.5)
        plt.title("Arrival axis by file time")
        plt.xlabel("Time from file name")
        plt.ylabel("Arrival axis, deg")
        plt.ylim(0, 180)
        plt.grid(alpha=0.3)
        plt.gcf().autofmt_xdate()
        plt.savefig(output_dir / "azimuth_by_file_time.png", dpi=dpi, bbox_inches="tight")
        plt.close()


def plot_direction_map(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    receiver_lat: float,
    receiver_lon: float,
    radius_km: float,
    dpi: int,
) -> None:
    plt = configure_matplotlib()
    good = [row for row in rows if row["quality_flag"] == "good"]
    if not good:
        return

    angles = np.asarray([float(row["arrival_axis_deg"]) for row in good], dtype=float)
    hist, edges = np.histogram(angles, bins=np.arange(0, 181, 10))
    peak_index = int(np.argmax(hist))
    peak_angle = float((edges[peak_index] + edges[peak_index + 1]) / 2.0)
    mean_angle = circular_axis_mean_180(angles)

    peak_side_1 = destination_point(receiver_lat, receiver_lon, peak_angle, radius_km)
    peak_side_2 = destination_point(receiver_lat, receiver_lon, (peak_angle + 180.0) % 360.0, radius_km)
    mean_side_1 = destination_point(receiver_lat, receiver_lon, mean_angle, radius_km * 0.82)
    mean_side_2 = destination_point(receiver_lat, receiver_lon, (mean_angle + 180.0) % 360.0, radius_km * 0.82)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(
        [peak_side_2[1], receiver_lon, peak_side_1[1]],
        [peak_side_2[0], receiver_lat, peak_side_1[0]],
        color="#d62728",
        linewidth=3.0,
        label=f"dominant bearing axis: {peak_angle:.0f}/{(peak_angle + 180.0) % 360.0:.0f} deg",
    )
    ax.plot(
        [mean_side_2[1], receiver_lon, mean_side_1[1]],
        [mean_side_2[0], receiver_lat, mean_side_1[0]],
        color="#1f77b4",
        linewidth=2.0,
        linestyle="--",
        label=f"mean reliable axis: {mean_angle:.1f}/{(mean_angle + 180.0) % 360.0:.1f} deg",
    )
    ax.scatter([receiver_lon], [receiver_lat], s=90, color="black", zorder=5)
    ax.text(receiver_lon + 0.18, receiver_lat + 0.12, "Irkutsk receiver", fontsize=10, weight="bold")

    labels = [
        ("Lake Baikal", 107.0, 53.3),
        ("Ulan-Ude", 107.6, 51.8),
        ("Angarsk", 103.9, 52.5),
        ("Sayan region", 100.5, 51.2),
        ("Transbaikalia", 111.0, 51.8),
    ]
    for label, lon, lat in labels:
        ax.scatter([lon], [lat], s=26, color="#666666", alpha=0.75)
        ax.text(lon + 0.12, lat + 0.08, label, fontsize=9, color="#444444")

    all_lats = [receiver_lat, peak_side_1[0], peak_side_2[0], mean_side_1[0], mean_side_2[0]]
    all_lons = [receiver_lon, peak_side_1[1], peak_side_2[1], mean_side_1[1], mean_side_2[1]]
    margin_lat = 0.9
    margin_lon = 1.2
    ax.set_xlim(min(all_lons) - margin_lon, max(all_lons) + margin_lon)
    ax.set_ylim(min(all_lats) - margin_lat, max(all_lats) + margin_lat)
    ax.set_xlabel("Longitude, deg")
    ax.set_ylabel("Latitude, deg")
    ax.set_title("Experimental sferic arrival bearing from Irkutsk\nsingle-station axis, 180-degree ambiguity")
    ax.grid(alpha=0.35)
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.02,
        "This map shows a bearing line only, not source coordinates or range.",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    fig.savefig(output_dir / "azimuth_direction_map.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_examples(examples: list[tuple[dict[str, Any], dict[str, np.ndarray]]], output_dir: Path, dpi: int) -> None:
    plt = configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (row, diag) in enumerate(examples, start=1):
        ns = diag["ns"]
        we = diag["we"]
        time_ms = diag["time_ms"]
        principal = diag["principal"]
        scale = max(float(np.std(ns)), float(np.std(we)), 1e-9) * 3.0
        axis_x = np.array([-principal[0], principal[0]]) * scale
        axis_y = np.array([-principal[1], principal[1]]) * scale

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        axes[0].plot(time_ms, ns, label="ns", linewidth=1.0)
        axes[0].plot(time_ms, we, label="we", linewidth=1.0)
        axes[0].set_title("Waveforms in event window")
        axes[0].set_xlabel("Time, ms")
        axes[0].set_ylabel("Filtered amplitude")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].scatter(ns, we, s=4, alpha=0.35)
        axes[1].plot(axis_x, axis_y, color="red", linewidth=2.0, label="PCA axis")
        axes[1].set_title("Hodogram ns vs we")
        axes[1].set_xlabel("ns")
        axes[1].set_ylabel("we")
        axes[1].axis("equal")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.suptitle(
            f"arrival={float(row['arrival_axis_deg']):.1f} deg, "
            f"candidates={float(row['azimuth_candidate_1']):.1f}/{float(row['azimuth_candidate_2']):.1f} deg, "
            f"linearity={float(row['pca_linearity']):.1f}"
        )
        fig.savefig(output_dir / f"event_{index:03d}_azimuth_diagnostic.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    report = output_dir / "AZIMUTH_ESTIMATION_REPORT.md"
    peak_text = "нет надежных событий"
    if "peak_bin_start_deg" in summary:
        peak_text = (
            f"{summary['peak_bin_start_deg']:.0f}-{summary['peak_bin_end_deg']:.0f} degrees "
            f"({summary['peak_bin_count']} events)"
        )
    report.write_text(
        f"""# Experimental Sferic Arrival-Axis Estimation

This report summarizes an experimental single-station azimuth analysis based on two orthogonal loop antenna channels, `ns` and `we`.

## Summary

- processed events: `{summary['events_processed']}`
- reliable events: `{summary['reliable_events']}`
- dominant arrival-axis bin: `{peak_text}`
- mean arrival axis for reliable events: `{summary.get('mean_arrival_axis_deg', 'N/A')}`

The result is an **arrival line / bearing axis**, not a source coordinate. Because a single station with two orthogonal loop antennas has a 180-degree ambiguity, every result is represented by two possible azimuth candidates:

```text
azimuth_candidate_1 = arrival_axis_deg
azimuth_candidate_2 = arrival_axis_deg + 180 degrees
```

## Generated Plots

- `azimuth_histogram.png`
- `azimuth_rose.png`
- `ratio_we_ns_histogram.png`
- `pca_linearity_histogram.png`
- `azimuth_by_file_time.png`
- `azimuth_direction_map.png`
- `examples/`

## Interpretation

The peak of the histogram indicates the most common estimated line of arrival for the analyzed events. The opposite direction is physically equivalent in this preliminary single-station estimate because of the 180-degree ambiguity.

Distance to the lightning discharge is **not** estimated by this method.

The map figure uses Irkutsk as the receiver position and draws the dominant bearing axis plus the opposite candidate. It is a geographic schematic of the line of arrival only, not a localization result.

## Quality Flags

Quality counts:

```json
{json.dumps(summary.get('quality_counts', {}), indent=2, ensure_ascii=False)}
```

Events with low energy, low SNR, weak PCA linearity, saturation, or a single-channel detection are retained in the CSV but marked as unreliable via `quality_flag`.

## Method Limitations

- `ns` and `we` channels currently do not have precise amplitude-phase calibration.
- Loop antennas have a figure-eight directional pattern.
- A single station gives a 180-degree ambiguity.
- Absolute amplitude does not determine distance to the source.
- The estimated azimuth should be compared with external lightning-location data, for example Blitzortung or WWLLN.
- Full source localization requires a network of multiple synchronized receivers.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    plots_dir = output_dir
    examples_dir = output_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    broadband = import_project_reader()
    signal_mod = import_scipy_signal() if not args.no_filter else None
    bandpass_sos = None
    if signal_mod is not None:
        bandpass_sos = signal_mod.butter(
            6,
            [args.freq_min, args.freq_max],
            btype="bandpass",
            fs=args.sample_rate,
            output="sos",
        )

    detections = load_event_detections(args.events_csv.expanduser().resolve(), args.max_events)
    merged_events = merge_detections(detections, args.merge_tolerance)
    analysis_window_s = args.analysis_window_ms / 1000.0
    filter_window_s = max(args.filter_window_ms / 1000.0, analysis_window_s)

    events_by_file: dict[Path, list[MergedEvent]] = defaultdict(list)
    for event in merged_events:
        events_by_file[event.source_file].append(event)

    results: list[dict[str, Any]] = []
    examples: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    for file_index, (source_file, file_events) in enumerate(sorted(events_by_file.items(), key=lambda item: str(item[0])), start=1):
        print(f"[{file_index}/{len(events_by_file)}] {source_file.name} events={len(file_events)}", flush=True)
        record_size, sample_count, data_offset = broadband.read_metadata(source_file)
        samples = broadband.load_samples(source_file, sample_count, record_size, data_offset)
        ns_samples = broadband.extract_channel(samples, sample_count, record_size, args.channels, CHANNEL_INDEX["ns"])
        we_samples = broadband.extract_channel(samples, sample_count, record_size, args.channels, CHANNEL_INDEX["we"])
        for event in file_events:
            try:
                row, diagnostic = estimate_event(
                    event,
                    ns_samples,
                    we_samples,
                    sample_rate=args.sample_rate,
                    analysis_window_s=analysis_window_s,
                    filter_window_s=filter_window_s,
                    bandpass_sos=bandpass_sos,
                    signal_mod=signal_mod,
                    gain_ns=args.gain_ns,
                    gain_we=args.gain_we,
                    phase_shift_samples=args.phase_shift_samples,
                    antenna_angle_offset_deg=args.antenna_angle_offset_deg,
                    min_energy=args.min_energy,
                    min_snr=args.min_snr,
                    min_linearity=args.min_linearity,
                    saturation_level=args.saturation_level,
                )
            except Exception as exc:
                row = {
                    "source_file": str(event.source_file),
                    "event_time_s": event.event_time_s,
                    "event_datetime_utc_if_available": "",
                    "detected_channels": "|".join(event.channels),
                    "merged_detection_count": int(event.detections),
                    "A_ns": "",
                    "A_we": "",
                    "ratio_we_ns": "",
                    "energy_ns": "",
                    "energy_we": "",
                    "snr_ns": "",
                    "snr_we": "",
                    "magnetic_axis_deg": "",
                    "arrival_axis_deg": "",
                    "azimuth_candidate_1": "",
                    "azimuth_candidate_2": "",
                    "pca_lambda1": "",
                    "pca_lambda2": "",
                    "pca_linearity": "",
                    "is_saturated": "",
                    "quality_flag": f"error:{exc}",
                }
                diagnostic = None
            results.append(row)
            if diagnostic is not None and row["quality_flag"] == "good" and len(examples) < args.example_count:
                examples.append((row, diagnostic))

    fieldnames = [
        "source_file",
        "event_time_s",
        "event_datetime_utc_if_available",
        "detected_channels",
        "merged_detection_count",
        "A_ns",
        "A_we",
        "ratio_we_ns",
        "energy_ns",
        "energy_we",
        "snr_ns",
        "snr_we",
        "magnetic_axis_deg",
        "arrival_axis_deg",
        "azimuth_candidate_1",
        "azimuth_candidate_2",
        "pca_lambda1",
        "pca_lambda2",
        "pca_linearity",
        "is_saturated",
        "quality_flag",
    ]
    csv_path = output_dir / "sferic_azimuth_estimates.csv"
    write_csv_rows(csv_path, results, fieldnames)
    summary = summarize_results(results)
    (output_dir / "azimuth_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    make_plots(results, plots_dir, args.plot_dpi)
    plot_direction_map(
        results,
        plots_dir,
        receiver_lat=args.receiver_lat,
        receiver_lon=args.receiver_lon,
        radius_km=args.map_radius_km,
        dpi=args.plot_dpi,
    )
    plot_examples(examples, examples_dir, args.plot_dpi)
    write_report(output_dir, summary)

    print(f"Detections loaded : {len(detections)}")
    print(f"Merged events     : {len(merged_events)}")
    print(f"Results CSV       : {csv_path}")
    print(f"Reliable events   : {summary['reliable_events']}")
    print(f"Report            : {output_dir / 'AZIMUTH_ESTIMATION_REPORT.md'}")


if __name__ == "__main__":
    main()
