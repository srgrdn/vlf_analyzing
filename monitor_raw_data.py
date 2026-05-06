#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


STATE_FILE_NAME = "monitor_state.json"
LOG_FILE_NAME = "monitor.log"
DEFAULT_PATTERN = "Broadband_Data_*.bin"


@dataclass
class MonitorConfig:
    watch_dir: Path
    state_path: Path
    log_path: Path
    detector_script: Path
    python_executable: str
    poll_interval: float
    stable_cycles: int
    channel: str
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
    hits_dir: Path | None
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
    parser.add_argument("--channel", default="ns", help="Detector channel. Default: ns")
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
        help="Directory for files with zero detections. Default: <watch-dir>/empty unless --delete-empty is used.",
    )
    parser.add_argument(
        "--hits-dir",
        type=Path,
        default=None,
        help="Optional directory where files with detections are moved. Default: leave in place.",
    )
    parser.add_argument(
        "--failed-dir",
        type=Path,
        default=None,
        help="Optional directory where files with detector errors are moved. Default: <watch-dir>/failed",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> MonitorConfig:
    watch_dir = args.watch_dir.expanduser().resolve()
    monitor_dir = watch_dir / ".monitor"
    state_path = args.state_file.expanduser().resolve() if args.state_file else monitor_dir / STATE_FILE_NAME
    log_path = args.log_file.expanduser().resolve() if args.log_file else monitor_dir / LOG_FILE_NAME
    failed_dir = args.failed_dir.expanduser().resolve() if args.failed_dir else watch_dir / "failed"
    empty_dir = None if args.delete_empty else (
        args.empty_dir.expanduser().resolve() if args.empty_dir else watch_dir / "empty"
    )
    hits_dir = args.hits_dir.expanduser().resolve() if args.hits_dir else None
    detector_script = args.detector_script.expanduser().resolve()
    return MonitorConfig(
        watch_dir=watch_dir,
        state_path=state_path,
        log_path=log_path,
        detector_script=detector_script,
        python_executable=sys.executable,
        poll_interval=args.poll_interval,
        stable_cycles=args.stable_cycles,
        channel=args.channel,
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
        hits_dir=hits_dir,
        failed_dir=failed_dir,
        detector_timeout=args.detector_timeout,
        file_pattern=args.file_pattern,
    )


def ensure_runtime_dirs(config: MonitorConfig) -> None:
    config.watch_dir.mkdir(parents=True, exist_ok=True)
    config.state_path.parent.mkdir(parents=True, exist_ok=True)
    if config.empty_dir is not None:
        config.empty_dir.mkdir(parents=True, exist_ok=True)
    if config.hits_dir is not None:
        config.hits_dir.mkdir(parents=True, exist_ok=True)
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


def build_detector_command(config: MonitorConfig, input_path: Path) -> list[str]:
    return [
        config.python_executable,
        str(config.detector_script),
        str(input_path),
        "--channel",
        config.channel,
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


def parse_total_detections(stdout: str) -> int:
    for line in reversed(stdout.splitlines()):
        if line.startswith("TOTAL_DETECTIONS="):
            return int(line.split("=", 1)[1].strip())
    raise ValueError("Detector output did not include TOTAL_DETECTIONS line.")


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


def finalize_file(
    source: Path,
    detections: int,
    config: MonitorConfig,
    state: dict[str, Any],
) -> tuple[str, str]:
    if detections > 0:
        if config.hits_dir is None:
            return "hit", str(source)
        destination = unique_destination(config.hits_dir, source.name)
        source.replace(destination)
        return "hit", str(destination)

    if config.delete_empty:
        source.unlink(missing_ok=False)
        return "empty_deleted", ""

    assert config.empty_dir is not None
    destination = unique_destination(config.empty_dir, source.name)
    source.replace(destination)
    return "empty_moved", str(destination)


def move_failed(source: Path, config: MonitorConfig) -> str:
    assert config.failed_dir is not None
    destination = unique_destination(config.failed_dir, source.name)
    source.replace(destination)
    return str(destination)


def process_file(path: Path, config: MonitorConfig, state: dict[str, Any]) -> None:
    key = str(path)
    state["in_progress"][key] = {"started_at": utc_now()}
    command = build_detector_command(config, path)
    append_log(config.log_path, f"Processing {path.name}")

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
            "error": str(exc),
            "final_path": destination,
        }
        state["in_progress"].pop(key, None)
        state["observed"].pop(key, None)
        append_log(config.log_path, f"Failed to start detector for {path.name}: {exc}")
        return

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()

    if completed.returncode != 0:
        destination = move_failed(path, config) if path.exists() else ""
        state["processed"][key] = {
            "status": "detector_error",
            "processed_at": utc_now(),
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "final_path": destination,
        }
        state["in_progress"].pop(key, None)
        state["observed"].pop(key, None)
        append_log(config.log_path, f"Detector failed for {path.name} with code {completed.returncode}")
        if stdout:
            append_log(config.log_path, f"stdout: {stdout}")
        if stderr:
            append_log(config.log_path, f"stderr: {stderr}")
        return

    try:
        detections = parse_total_detections(stdout)
    except Exception as exc:
        destination = move_failed(path, config) if path.exists() else ""
        state["processed"][key] = {
            "status": "parse_error",
            "processed_at": utc_now(),
            "error": str(exc),
            "stdout": stdout,
            "stderr": stderr,
            "final_path": destination,
        }
        state["in_progress"].pop(key, None)
        state["observed"].pop(key, None)
        append_log(config.log_path, f"Could not parse detector result for {path.name}: {exc}")
        return

    status, final_path = finalize_file(path, detections, config, state)
    state["processed"][key] = {
        "status": status,
        "processed_at": utc_now(),
        "detections": detections,
        "final_path": final_path,
        "stdout": stdout,
    }
    state["in_progress"].pop(key, None)
    state["observed"].pop(key, None)
    append_log(config.log_path, f"Finished {path.name}: detections={detections}, status={status}")


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
