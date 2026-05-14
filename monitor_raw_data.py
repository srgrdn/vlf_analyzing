#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


STATE_FILE_NAME = "monitor_state.json"
LOG_FILE_NAME = "monitor.log"
DEFAULT_PATTERN = "Broadband_Data_*.bin"


@dataclass
class MonitorConfig:
    watch_dir: Path
    root_dir: Path
    state_path: Path
    log_path: Path
    reports_dir: Path
    per_channel_report: Path
    summary_report: Path
    events_report: Path
    detector_script: Path
    python_executable: str
    poll_interval: float
    stable_cycles: int
    freq_min: float
    freq_max: float
    segment_duration: float
    time_resolution: float
    frequency_resolution: float
    threshold_mad: float
    mark_span: float
    channels: int
    min_peak_distance: float
    delete_empty: bool
    empty_dir: Path | None
    processing1_dir: Path
    processing2_dir: Path
    use_processing2: bool
    failed_dir: Path | None
    detector_timeout: float | None
    file_pattern: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously monitor a directory with Broadband_Data_*.bin files, run the "
            "burst detector for each completed file, and remove or move files with no detections."
        )
    )
    parser.add_argument(
        "--watch-dir",
        type=Path,
        default=Path("raw_data"),
        help="Directory to monitor. Default: ./raw_data",
    )
    parser.add_argument(
        "--detector-script",
        type=Path,
        default=Path(__file__).resolve().with_name("detect_broadband_bursts.py"),
        help="Path to detect_broadband_bursts.py. Default: sibling file",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Path to monitor state JSON. Default: <watch-dir>/.monitor/monitor_state.json",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to monitor log file. Default: <watch-dir>/.monitor/monitor.log",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Seconds between directory scans. Default: 10",
    )
    parser.add_argument(
        "--stable-cycles",
        type=int,
        default=2,
        help="How many unchanged scans are required before a file is treated as complete. Default: 2",
    )
    parser.add_argument(
        "--detector-timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds for one detector run.",
    )
    parser.add_argument(
        "--file-pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob pattern for input files. Default: {DEFAULT_PATTERN}",
    )
    parser.add_argument("--channels", type=int, default=2, help="Detector channel count. Default: 2")
    parser.add_argument("--freq-min", type=float, default=20000.0, help="Detector lower frequency bound in Hz.")
    parser.add_argument("--freq-max", type=float, default=30000.0, help="Detector upper frequency bound in Hz.")
    parser.add_argument("--segment-duration", type=float, default=60.0, help="Detector segment duration in seconds.")
    parser.add_argument("--time-resolution", type=float, default=0.002, help="Detector time step in seconds.")
    parser.add_argument(
        "--frequency-resolution",
        type=float,
        default=50.0,
        help="Detector frequency step target in Hz.",
    )
    parser.add_argument("--threshold-mad", type=float, default=8.0, help="Detector threshold MAD multiplier.")
    parser.add_argument("--mark-span", type=float, default=0.01, help="Detector highlight span in seconds.")
    parser.add_argument(
        "--min-peak-distance",
        type=float,
        default=0.01,
        help="Detector minimum distance between peaks in seconds.",
    )
    parser.add_argument(
        "--delete-empty",
        action="store_true",
        help="Delete files with zero detections. If not set, they are moved to --empty-dir.",
    )
    parser.add_argument(
        "--empty-dir",
        type=Path,
        default=None,
        help="Directory for files with zero detections. Default: sibling ./empty unless --delete-empty is used.",
    )
    parser.add_argument(
        "--processing1-dir",
        type=Path,
        default=None,
        help="Directory for non-empty files used in the sferics pipeline. Default: sibling ./processing1",
    )
    parser.add_argument(
        "--processing2-dir",
        type=Path,
        default=None,
        help="Directory for non-empty files used in the whistler pipeline. Default: sibling ./processing2",
    )
    parser.add_argument(
        "--disable-processing2",
        action="store_true",
        help="Do not create or use processing2. Non-empty files will only be moved to processing1.",
    )
    parser.add_argument(
        "--failed-dir",
        type=Path,
        default=None,
        help="Directory where files with detector errors are moved. Default: sibling ./failed",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> MonitorConfig:
    watch_dir = args.watch_dir.expanduser().resolve()
    root_dir = watch_dir.parent
    monitor_dir = watch_dir / ".monitor"
    reports_dir = root_dir / "reports"
    state_path = args.state_file.expanduser().resolve() if args.state_file else monitor_dir / STATE_FILE_NAME
    log_path = args.log_file.expanduser().resolve() if args.log_file else monitor_dir / LOG_FILE_NAME
    failed_dir = args.failed_dir.expanduser().resolve() if args.failed_dir else root_dir / "failed"
    empty_dir = None if args.delete_empty else (
        args.empty_dir.expanduser().resolve() if args.empty_dir else root_dir / "empty"
    )
    processing1_dir = (
        args.processing1_dir.expanduser().resolve()
        if args.processing1_dir else root_dir / "processing1"
    )
    processing2_dir = (
        args.processing2_dir.expanduser().resolve()
        if args.processing2_dir else root_dir / "processing2"
    )
    detector_script = args.detector_script.expanduser().resolve()
    return MonitorConfig(
        watch_dir=watch_dir,
        root_dir=root_dir,
        state_path=state_path,
        log_path=log_path,
        reports_dir=reports_dir,
        per_channel_report=reports_dir / "sferics_per_channel.csv",
        summary_report=reports_dir / "sferics_summary.csv",
        events_report=reports_dir / "sferics_events.csv",
        detector_script=detector_script,
        python_executable=sys.executable,
        poll_interval=args.poll_interval,
        stable_cycles=args.stable_cycles,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        segment_duration=args.segment_duration,
        time_resolution=args.time_resolution,
        frequency_resolution=args.frequency_resolution,
        threshold_mad=args.threshold_mad,
        mark_span=args.mark_span,
        channels=args.channels,
        min_peak_distance=args.min_peak_distance,
        delete_empty=args.delete_empty,
        empty_dir=empty_dir,
        processing1_dir=processing1_dir,
        processing2_dir=processing2_dir,
        use_processing2=not args.disable_processing2,
        failed_dir=failed_dir,
        detector_timeout=args.detector_timeout,
        file_pattern=args.file_pattern,
    )


def ensure_runtime_dirs(config: MonitorConfig) -> None:
    config.watch_dir.mkdir(parents=True, exist_ok=True)
    config.state_path.parent.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    if config.empty_dir is not None:
        config.empty_dir.mkdir(parents=True, exist_ok=True)
    config.processing1_dir.mkdir(parents=True, exist_ok=True)
    if config.use_processing2:
        config.processing2_dir.mkdir(parents=True, exist_ok=True)
    if config.failed_dir is not None:
        config.failed_dir.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def append_log(log_path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + os.linesep)


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {
            "processed": {},
            "in_progress": {},
            "observed": {},
        }
    with state_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("processed", {})
    data.setdefault("in_progress", {})
    data.setdefault("observed", {})
    return data


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False, sort_keys=True)
    tmp_path.replace(state_path)


def cleanup_missing_entries(state: dict[str, Any], existing_paths: set[str]) -> None:
    for key in ("observed", "in_progress"):
        stale = [path for path in state[key] if path not in existing_paths]
        for path in stale:
            state[key].pop(path, None)


def is_already_processed(state: dict[str, Any], path: Path) -> bool:
    return str(path) in state["processed"]


def mark_observed(state: dict[str, Any], path: Path, stat_result: os.stat_result, stable_cycles: int) -> bool:
    key = str(path)
    previous = state["observed"].get(key)
    size = int(stat_result.st_size)
    mtime_ns = int(getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)))

    if previous is None:
        state["observed"][key] = {
            "size": size,
            "mtime_ns": mtime_ns,
            "stable_count": 0,
            "first_seen_at": utc_now(),
        }
        return False

    unchanged = previous["size"] == size and previous["mtime_ns"] == mtime_ns
    previous["size"] = size
    previous["mtime_ns"] = mtime_ns
    previous["stable_count"] = previous.get("stable_count", 0) + 1 if unchanged else 0
    previous["last_seen_at"] = utc_now()
    return previous["stable_count"] >= stable_cycles


def build_detector_command(config: MonitorConfig, input_path: Path, channel: str) -> list[str]:
    return [
        config.python_executable,
        str(config.detector_script),
        str(input_path),
        "--channel",
        channel,
        "--channels",
        str(config.channels),
        "--freq-min",
        str(config.freq_min),
        "--freq-max",
        str(config.freq_max),
        "--segment-duration",
        str(config.segment_duration),
        "--time-resolution",
        str(config.time_resolution),
        "--frequency-resolution",
        str(config.frequency_resolution),
        "--threshold-mad",
        str(config.threshold_mad),
        "--mark-span",
        str(config.mark_span),
        "--min-peak-distance",
        str(config.min_peak_distance),
        "--no-save",
        "--allow-no-output",
    ]


def parse_detection_summary(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("DETECTION_SUMMARY_JSON="):
            return json.loads(line.split("=", 1)[1].strip())
    raise ValueError("Detector output did not include DETECTION_SUMMARY_JSON line.")


def parse_detection_events(stdout: str) -> list[dict[str, Any]]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("DETECTION_EVENTS_JSON="):
            return json.loads(line.split("=", 1)[1].strip())
    raise ValueError("Detector output did not include DETECTION_EVENTS_JSON line.")


def parse_file_datetime(file_name: str) -> datetime | None:
    prefix = "Broadband_Data_"
    suffix = ".bin"
    if not file_name.startswith(prefix) or not file_name.endswith(suffix):
        return None
    payload = file_name[len(prefix):-len(suffix)]
    try:
        return datetime.strptime(payload, "%Y.%m.%d_%H.%M.%S")
    except ValueError:
        return None


def unique_destination(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        alternative = directory / f"{stem}_{index:03d}{suffix}"
        if not alternative.exists():
            return alternative
        index += 1


def append_csv_row(output_path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def finalize_file(
    source: Path,
    total_detections: int,
    config: MonitorConfig,
    summaries: dict[str, dict[str, Any]],
) -> dict[str, str]:
    if total_detections > 0:
        processing1_path = unique_destination(config.processing1_dir, source.name)
        source.replace(processing1_path)

        processing2_path = ""
        if config.use_processing2:
            destination = unique_destination(config.processing2_dir, source.name)
            shutil.copy2(processing1_path, destination)
            processing2_path = str(destination)

        dominant_channel = max(
            summaries,
            key=lambda channel_name: int(summaries[channel_name]["total_detections"]),
        )
        return {
            "status": "routed",
            "final_path": str(processing1_path),
            "processing1_path": str(processing1_path),
            "processing2_path": processing2_path,
            "dominant_channel": dominant_channel,
        }

    if config.delete_empty:
        source.unlink(missing_ok=False)
        return {
            "status": "empty_deleted",
            "final_path": "",
            "processing1_path": "",
            "processing2_path": "",
            "dominant_channel": "none",
        }

    assert config.empty_dir is not None
    destination = unique_destination(config.empty_dir, source.name)
    source.replace(destination)
    return {
        "status": "empty_moved",
        "final_path": str(destination),
        "processing1_path": "",
        "processing2_path": "",
        "dominant_channel": "none",
    }


def move_failed(source: Path, config: MonitorConfig) -> str:
    assert config.failed_dir is not None
    destination = unique_destination(config.failed_dir, source.name)
    source.replace(destination)
    return str(destination)


def write_reports(
    config: MonitorConfig,
    path: Path,
    file_dt: datetime | None,
    channel_summaries: dict[str, dict[str, Any]],
    channel_events: dict[str, list[dict[str, Any]]],
    routing: dict[str, str],
) -> None:
    file_time = file_dt.isoformat(sep=" ") if file_dt is not None else ""
    per_channel_fields = [
        "file_name",
        "file_time",
        "channel",
        "detections_count",
        "max_peak_db",
        "mean_peak_db",
        "median_peak_db",
        "status",
        "final_path",
    ]
    for channel_name, summary in channel_summaries.items():
        append_csv_row(
            config.per_channel_report,
            per_channel_fields,
            {
                "file_name": path.name,
                "file_time": file_time or "",
                "channel": channel_name,
                "detections_count": int(summary["total_detections"]),
                "max_peak_db": summary["max_peak_db"] if summary["max_peak_db"] is not None else "",
                "mean_peak_db": summary["mean_peak_db"] if summary["mean_peak_db"] is not None else "",
                "median_peak_db": summary["median_peak_db"] if summary["median_peak_db"] is not None else "",
                "status": routing["status"],
                "final_path": routing["final_path"],
            },
        )

    event_fields = [
        "file_name",
        "file_time",
        "channel",
        "peak_time_local_s",
        "peak_time_absolute",
        "peak_time_global_s",
        "peak_db",
        "threshold_db",
        "segment_index",
        "segment_start_s",
        "segment_stop_s",
    ]
    for channel_name, events in channel_events.items():
        for event in events:
            peak_time_absolute = ""
            if file_dt is not None:
                peak_dt = file_dt + timedelta(seconds=float(event["peak_time_global_s"]))
                peak_time_absolute = peak_dt.isoformat(sep=" ")
            append_csv_row(
                config.events_report,
                event_fields,
                {
                    "file_name": path.name,
                    "file_time": file_time,
                    "channel": channel_name,
                    "peak_time_local_s": float(event["peak_time_local_s"]),
                    "peak_time_absolute": peak_time_absolute,
                    "peak_time_global_s": float(event["peak_time_global_s"]),
                    "peak_db": float(event["peak_db"]),
                    "threshold_db": float(event["threshold_db"]),
                    "segment_index": int(event["segment_index"]),
                    "segment_start_s": float(event["segment_start_s"]),
                    "segment_stop_s": float(event["segment_stop_s"]),
                },
            )

    count_ns = int(channel_summaries["ns"]["total_detections"])
    count_we = int(channel_summaries["we"]["total_detections"])
    max_level_ns = channel_summaries["ns"]["max_peak_db"]
    max_level_we = channel_summaries["we"]["max_peak_db"]
    summary_fields = [
        "file_name",
        "file_time",
        "count_ns",
        "count_we",
        "count_total",
        "max_level_ns",
        "max_level_we",
        "dominant_channel",
        "is_empty",
        "final_route",
        "processing1_path",
        "processing2_path",
    ]
    append_csv_row(
        config.summary_report,
        summary_fields,
        {
            "file_name": path.name,
            "file_time": file_time or "",
            "count_ns": count_ns,
            "count_we": count_we,
            "count_total": count_ns + count_we,
            "max_level_ns": max_level_ns if max_level_ns is not None else "",
            "max_level_we": max_level_we if max_level_we is not None else "",
            "dominant_channel": routing["dominant_channel"],
            "is_empty": int((count_ns + count_we) == 0),
            "final_route": routing["status"],
            "processing1_path": routing["processing1_path"],
            "processing2_path": routing["processing2_path"],
        },
    )


def process_file(path: Path, config: MonitorConfig, state: dict[str, Any]) -> None:
    key = str(path)
    state["in_progress"][key] = {"started_at": utc_now()}
    append_log(config.log_path, f"Processing {path.name}")
    file_dt = parse_file_datetime(path.name)
    channel_summaries: dict[str, dict[str, Any]] = {}
    channel_events: dict[str, list[dict[str, Any]]] = {}

    for channel_name in ("ns", "we"):
        command = build_detector_command(config, path, channel_name)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=config.detector_timeout,
                check=False,
            )
        except Exception as exc:
            destination = move_failed(path, config) if path.exists() else ""
            state["processed"][key] = {
                "status": "failed_to_start",
                "processed_at": utc_now(),
                "channel": channel_name,
                "error": str(exc),
                "final_path": destination,
            }
            state["in_progress"].pop(key, None)
            state["observed"].pop(key, None)
            append_log(config.log_path, f"Failed to start detector for {path.name} ({channel_name}): {exc}")
            return

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if completed.returncode != 0:
            destination = move_failed(path, config) if path.exists() else ""
            state["processed"][key] = {
                "status": "detector_error",
                "processed_at": utc_now(),
                "channel": channel_name,
                "returncode": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "final_path": destination,
            }
            state["in_progress"].pop(key, None)
            state["observed"].pop(key, None)
            append_log(config.log_path, f"Detector failed for {path.name} ({channel_name}) with code {completed.returncode}")
            if stdout:
                append_log(config.log_path, f"stdout: {stdout}")
            if stderr:
                append_log(config.log_path, f"stderr: {stderr}")
            return

        try:
            channel_summaries[channel_name] = parse_detection_summary(stdout)
            channel_events[channel_name] = parse_detection_events(stdout)
        except Exception as exc:
            destination = move_failed(path, config) if path.exists() else ""
            state["processed"][key] = {
                "status": "parse_error",
                "processed_at": utc_now(),
                "channel": channel_name,
                "error": str(exc),
                "stdout": stdout,
                "stderr": stderr,
                "final_path": destination,
            }
            state["in_progress"].pop(key, None)
            state["observed"].pop(key, None)
            append_log(config.log_path, f"Could not parse detector result for {path.name} ({channel_name}): {exc}")
            return

    total_detections = sum(int(summary["total_detections"]) for summary in channel_summaries.values())
    routing = finalize_file(
        source=path,
        total_detections=total_detections,
        config=config,
        summaries=channel_summaries,
    )
    write_reports(config, path, file_dt, channel_summaries, channel_events, routing)
    state["processed"][key] = {
        "status": routing["status"],
        "processed_at": utc_now(),
        "file_time": file_dt.isoformat(sep=" ") if file_dt is not None else None,
        "count_ns": int(channel_summaries["ns"]["total_detections"]),
        "count_we": int(channel_summaries["we"]["total_detections"]),
        "count_total": total_detections,
        "dominant_channel": routing["dominant_channel"],
        "final_path": routing["final_path"],
        "processing1_path": routing["processing1_path"],
        "processing2_path": routing["processing2_path"],
        "ns": channel_summaries["ns"],
        "we": channel_summaries["we"],
    }
    state["in_progress"].pop(key, None)
    state["observed"].pop(key, None)
    append_log(
        config.log_path,
        (
            f"Finished {path.name}: total={total_detections}, "
            f"ns={channel_summaries['ns']['total_detections']}, "
            f"we={channel_summaries['we']['total_detections']}, "
            f"status={routing['status']}"
        ),
    )


def monitor_loop(config: MonitorConfig) -> None:
    state = load_state(config.state_path)
    append_log(config.log_path, f"Watching {config.watch_dir}")
    append_log(config.log_path, f"Detector script: {config.detector_script}")

    while True:
        try:
            candidates = sorted(
                path for path in config.watch_dir.glob(config.file_pattern)
                if path.is_file()
            )
            existing_paths = {str(path) for path in candidates}
            cleanup_missing_entries(state, existing_paths)

            ready_files: list[Path] = []
            for path in candidates:
                if is_already_processed(state, path):
                    continue
                stat_result = path.stat()
                if mark_observed(state, path, stat_result, config.stable_cycles):
                    ready_files.append(path)

            save_state(config.state_path, state)

            for path in ready_files:
                if not path.exists():
                    continue
                process_file(path, config, state)
                save_state(config.state_path, state)
        except KeyboardInterrupt:
            append_log(config.log_path, "Monitor stopped by user.")
            save_state(config.state_path, state)
            raise
        except Exception as exc:
            append_log(config.log_path, f"Unexpected monitor error: {exc}")
            save_state(config.state_path, state)

        time.sleep(config.poll_interval)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    ensure_runtime_dirs(config)

    if config.poll_interval <= 0:
        raise ValueError("Poll interval must be positive.")
    if config.stable_cycles < 1:
        raise ValueError("Stable cycles must be at least 1.")
    if not config.detector_script.exists():
        raise FileNotFoundError(f"Detector script not found: {config.detector_script}")

    monitor_loop(config)


if __name__ == "__main__":
    main()
