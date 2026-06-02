#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from envelope_model_inference import (
    DEFAULT_BURST_THRESHOLD,
    DEFAULT_CHANNELS,
    DEFAULT_FILE_PATTERN,
    DEFAULT_FREQUENCY_RESOLUTION,
    DEFAULT_FREQ_MAX,
    DEFAULT_FREQ_MIN,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SEGMENT_DURATION,
    DEFAULT_TIME_RESOLUTION,
    analyze_file_with_envelope_model,
    append_csv_row,
    append_json_line,
    collect_input_files,
    load_envelope_model,
    segment_rows_to_dicts,
    summary_to_dict,
)


@dataclass
class PendingFileState:
    size: int
    mtime: float
    stable_cycles: int = 0


@dataclass
class ProcessedFileState:
    status: str
    burst_segment_count: int
    final_path: str
    processed_at: float


@dataclass
class MonitorState:
    pending: dict[str, PendingFileState] = field(default_factory=dict)
    processed: dict[str, ProcessedFileState] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "pending": {key: asdict(value) for key, value in self.pending.items()},
            "processed": {key: asdict(value) for key, value in self.processed.items()},
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "MonitorState":
        pending = {
            key: PendingFileState(**value)
            for key, value in dict(payload.get("pending", {})).items()
        }
        processed = {
            key: ProcessedFileState(**value)
            for key, value in dict(payload.get("processed", {})).items()
        }
        return cls(pending=pending, processed=processed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously watch a directory with Broadband_Data_*.bin files, "
            "run envelope_1d inference on each stable minute file, and route empty/non-empty files."
        )
    )
    parser.add_argument("--watch-dir", type=Path, required=True, help="Directory with incoming Broadband_Data_*.bin files.")
    parser.add_argument("--model", type=Path, required=True, help="Path to a .pt checkpoint such as models/envelope_1d_dataset_v2.pt.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subdirectories under --watch-dir.")
    parser.add_argument("--file-pattern", default=DEFAULT_FILE_PATTERN, help=f"Input filename glob. Default: {DEFAULT_FILE_PATTERN}")
    parser.add_argument("--processing1-dir", type=Path, default=None, help="Directory for non-empty files. Default: sibling processing1/")
    parser.add_argument("--failed-dir", type=Path, default=None, help="Directory for failed files. Default: sibling failed/")
    parser.add_argument("--empty-dir", type=Path, default=None, help="Directory for empty files when --move-empty is used. Default: sibling empty/")
    parser.add_argument("--reports-dir", type=Path, default=None, help="Directory for model CSV reports. Default: sibling reports/model_envelope_1d/")
    parser.add_argument("--move-empty", action="store_true", help="Move files without burst segments to empty/ instead of deleting them.")
    parser.add_argument("--poll-interval", type=float, default=10.0, help="Delay between scan cycles in seconds. Default: 10")
    parser.add_argument("--stable-cycles", type=int, default=2, help="How many unchanged scans are required before processing a file. Default: 2")
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--channel", choices=("ns", "we", "both"), default="ns", help="Model channel to analyze. Use both to run ns and we. Default: ns")
    parser.add_argument("--segment-duration", type=float, default=DEFAULT_SEGMENT_DURATION)
    parser.add_argument("--time-resolution", type=float, default=DEFAULT_TIME_RESOLUTION)
    parser.add_argument("--frequency-resolution", type=float, default=DEFAULT_FREQUENCY_RESOLUTION)
    parser.add_argument("--freq-min", type=float, default=DEFAULT_FREQ_MIN)
    parser.add_argument("--freq-max", type=float, default=DEFAULT_FREQ_MAX)
    parser.add_argument("--burst-threshold", type=float, default=DEFAULT_BURST_THRESHOLD)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    watch_dir = args.watch_dir.expanduser().resolve()
    base_dir = watch_dir.parent
    reports_dir = (args.reports_dir.expanduser().resolve() if args.reports_dir else base_dir / "reports" / "model_envelope_1d")
    processing1_dir = (args.processing1_dir.expanduser().resolve() if args.processing1_dir else base_dir / "processing1")
    failed_dir = (args.failed_dir.expanduser().resolve() if args.failed_dir else base_dir / "failed")
    empty_dir = (args.empty_dir.expanduser().resolve() if args.empty_dir else base_dir / "empty")
    monitor_dir = watch_dir / ".model_monitor"
    return {
        "watch_dir": watch_dir,
        "reports_dir": reports_dir,
        "processing1_dir": processing1_dir,
        "failed_dir": failed_dir,
        "empty_dir": empty_dir,
        "monitor_dir": monitor_dir,
        "segments_csv": reports_dir / "segments.csv",
        "summary_csv": reports_dir / "summary.csv",
        "state_json": monitor_dir / "monitor_state.json",
        "log_jsonl": monitor_dir / "monitor_events.jsonl",
    }


def ensure_dirs(paths: dict[str, Path], move_empty: bool) -> None:
    paths["watch_dir"].mkdir(parents=True, exist_ok=True)
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    paths["processing1_dir"].mkdir(parents=True, exist_ok=True)
    paths["failed_dir"].mkdir(parents=True, exist_ok=True)
    paths["monitor_dir"].mkdir(parents=True, exist_ok=True)
    if move_empty:
        paths["empty_dir"].mkdir(parents=True, exist_ok=True)


def load_state(state_path: Path) -> MonitorState:
    if not state_path.exists():
        return MonitorState()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return MonitorState.from_json(payload)


def save_state(state_path: Path, state: MonitorState) -> None:
    state_path.write_text(json.dumps(state.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")


def detect_stable_files(
    files: list[Path],
    state: MonitorState,
    stable_cycles_required: int,
) -> list[Path]:
    stable: list[Path] = []
    seen = {str(path.resolve()) for path in files}

    for missing in list(state.pending):
        if missing not in seen:
            del state.pending[missing]

    for path in files:
        resolved = str(path.resolve())
        stat = path.stat()
        info = state.pending.get(resolved)
        if info is None:
            state.pending[resolved] = PendingFileState(size=stat.st_size, mtime=stat.st_mtime, stable_cycles=0)
            continue

        if info.size == stat.st_size and info.mtime == stat.st_mtime:
            info.stable_cycles += 1
        else:
            info.size = stat.st_size
            info.mtime = stat.st_mtime
            info.stable_cycles = 0

        if info.stable_cycles >= stable_cycles_required:
            stable.append(path)

    return stable


def route_processed_file(
    file_path: Path,
    burst_segment_count: int,
    paths: dict[str, Path],
    move_empty: bool,
) -> tuple[str, Path | None]:
    if burst_segment_count > 0:
        target = paths["processing1_dir"] / file_path.name
        shutil.move(str(file_path), str(target))
        return "routed", target

    if move_empty:
        target = paths["empty_dir"] / file_path.name
        shutil.move(str(file_path), str(target))
        return "empty_moved", target

    file_path.unlink()
    return "empty_deleted", None


def process_file(
    file_path: Path,
    model,
    args: argparse.Namespace,
    paths: dict[str, Path],
    state: MonitorState,
) -> None:
    channels = ("ns", "we") if args.channel == "both" else (args.channel,)
    channel_summaries = {}
    pending_summary_rows = {}
    total_burst_segment_count = 0
    last_summary = None
    for channel in channels:
        per_segment, summary = analyze_file_with_envelope_model(
            file_path=file_path,
            model=model,
            channel=channel,
            sample_rate=args.sample_rate,
            channels=args.channels,
            segment_duration=args.segment_duration,
            time_resolution=args.time_resolution,
            frequency_resolution=args.frequency_resolution,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            burst_threshold=args.burst_threshold,
        )
        for row in segment_rows_to_dicts(per_segment):
            append_csv_row(paths["segments_csv"], row)
        channel_summaries[channel] = summary
        pending_summary_rows[channel] = summary
        total_burst_segment_count += summary.burst_segment_count
        last_summary = summary

    if last_summary is None:
        raise RuntimeError(f"No channel summaries were produced for {file_path}")

    if args.channel == "both":
        ns = channel_summaries["ns"]
        we = channel_summaries["we"]
        append_csv_row(
            paths["reports_dir"] / "summary_combined.csv",
            {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_time": ns.file_time or we.file_time,
                "duration_s": f"{max(ns.duration_s, we.duration_s):.6f}",
                "segment_count": max(ns.segment_count, we.segment_count),
                "burst_segment_count_ns": ns.burst_segment_count,
                "burst_segment_count_we": we.burst_segment_count,
                "burst_segment_count_total": ns.burst_segment_count + we.burst_segment_count,
                "max_prob_burst_ns": f"{ns.max_prob_burst:.6f}",
                "max_prob_burst_we": f"{we.max_prob_burst:.6f}",
                "mean_prob_burst_ns": f"{ns.mean_prob_burst:.6f}",
                "mean_prob_burst_we": f"{we.mean_prob_burst:.6f}",
                "dominant_channel": (
                    "ns"
                    if ns.burst_segment_count > we.burst_segment_count
                    else "we"
                    if we.burst_segment_count > ns.burst_segment_count
                    else "equal"
                ),
                "route_decision": "burst_detected" if (ns.burst_segment_count or we.burst_segment_count) else "empty",
            },
        )

    status, final_path = route_processed_file(
        file_path=file_path,
        burst_segment_count=total_burst_segment_count,
        paths=paths,
        move_empty=args.move_empty,
    )
    for channel in channels:
        append_csv_row(paths["summary_csv"], summary_to_dict(pending_summary_rows[channel], final_path=final_path))

    resolved = str(file_path.resolve())
    state.processed[resolved] = ProcessedFileState(
        status=status,
        burst_segment_count=total_burst_segment_count,
        final_path="" if final_path is None else str(final_path),
        processed_at=time.time(),
    )
    state.pending.pop(resolved, None)
    append_json_line(
        paths["log_jsonl"],
            {
                "timestamp": time.time(),
                "file_name": file_path.name,
                "status": status,
                "burst_segment_count": total_burst_segment_count,
                "route_decision": "burst_detected" if total_burst_segment_count else "empty",
                "final_path": "" if final_path is None else str(final_path),
            },
        )


def process_file_with_failure_handling(
    file_path: Path,
    model,
    args: argparse.Namespace,
    paths: dict[str, Path],
    state: MonitorState,
) -> None:
    try:
        process_file(file_path, model, args, paths, state)
    except Exception as exc:
        failed_target = paths["failed_dir"] / file_path.name
        if file_path.exists():
            shutil.move(str(file_path), str(failed_target))
        resolved = str(file_path.resolve())
        state.processed[resolved] = ProcessedFileState(
            status="failed",
            burst_segment_count=0,
            final_path=str(failed_target),
            processed_at=time.time(),
        )
        state.pending.pop(resolved, None)
        append_json_line(
            paths["log_jsonl"],
            {
                "timestamp": time.time(),
                "file_name": file_path.name,
                "status": "failed",
                "error": str(exc),
                "final_path": str(failed_target),
            },
        )


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    ensure_dirs(paths, move_empty=args.move_empty)
    state = load_state(paths["state_json"])
    model_kind, model = load_envelope_model(args.model)

    print(f"Loaded model    : {model_kind}")
    print(f"Watch dir       : {paths['watch_dir']}")
    print(f"Reports dir     : {paths['reports_dir']}")
    print(f"Processing1 dir : {paths['processing1_dir']}")
    print(f"Move empty      : {args.move_empty}")

    while True:
        if args.recursive:
            iterator = paths["watch_dir"].rglob(args.file_pattern)
        else:
            iterator = paths["watch_dir"].glob(args.file_pattern)
        files = sorted(path for path in iterator if path.is_file())
        files = [path for path in files if str(path.resolve()) not in state.processed]
        stable_files = detect_stable_files(files, state, stable_cycles_required=args.stable_cycles)
        if stable_files:
            stable_files.sort()
            for file_path in stable_files:
                print(f"Processing      : {file_path.name}")
                process_file_with_failure_handling(file_path, model, args, paths, state)
                save_state(paths["state_json"], state)
        else:
            save_state(paths["state_json"], state)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
