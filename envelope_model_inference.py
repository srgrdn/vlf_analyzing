#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from build_burst_dataset import compute_envelope_db
from plot_broadband_spectrogram import (
    CHANNEL_NAME_TO_INDEX,
    compute_spectrogram,
    crop_frequency_band,
    extract_channel,
    iter_time_segments,
    load_samples,
    read_metadata,
    resolve_frequency_band,
    resolve_stft_parameters,
)


DEFAULT_SAMPLE_RATE = 100_000.0
DEFAULT_CHANNELS = 2
DEFAULT_SEGMENT_DURATION = 2.0
DEFAULT_TIME_RESOLUTION = 0.002
DEFAULT_FREQUENCY_RESOLUTION = 50.0
DEFAULT_FREQ_MIN = 20_000.0
DEFAULT_FREQ_MAX = 30_000.0
DEFAULT_BURST_THRESHOLD = 0.5
DEFAULT_FILE_PATTERN = "Broadband_Data_*.bin"

LABEL_TO_INDEX = {"no_burst": 0, "burst": 1}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}

_FILE_TIME_RE = re.compile(r"(\d{4})\.(\d{2})\.(\d{2})_(\d{2})\.(\d{2})\.(\d{2})")


def ensure_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not available in the current interpreter. "
            "Install torch in this environment before running the envelope model pipeline."
        ) from exc
    return torch, nn


def collect_input_files(input_path: Path, recursive: bool, file_pattern: str = DEFAULT_FILE_PATTERN) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    iterator = input_path.rglob(file_pattern) if recursive else input_path.glob(file_pattern)
    files = sorted(path for path in iterator if path.is_file())
    if not files:
        raise ValueError(f"No files matching {file_pattern!r} found in {input_path}")
    return files


def parse_time_from_name(path: Path) -> str:
    match = _FILE_TIME_RE.search(path.name)
    if not match:
        return ""
    dt = datetime(
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
        int(match.group(5)),
        int(match.group(6)),
    )
    return dt.isoformat(sep=" ")


class EnvelopeCNNBase:
    @staticmethod
    def build():
        torch, nn = ensure_torch()

        class EnvelopeCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=9, padding=4),
                    nn.BatchNorm1d(16),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(16, 32, kernel_size=7, padding=3),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, kernel_size=5, padding=2),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 2))

            def forward(self, x):
                return self.classifier(self.features(x))

        return EnvelopeCNN()


def build_model_from_kind(model_kind: str):
    if model_kind.startswith("envelope_1d"):
        return EnvelopeCNNBase.build()
    raise ValueError(f"Unsupported model_kind={model_kind!r}. Expected an envelope_1d checkpoint.")


@dataclass
class SegmentPrediction:
    file_path: Path
    file_time: str
    channel: str
    segment_index: int
    segment_start: float
    segment_end: float
    predicted_label: str
    prob_no_burst: float
    prob_burst: float
    peak_value_db: float
    mean_value_db: float


@dataclass
class FilePredictionSummary:
    file_path: Path
    file_time: str
    channel: str
    duration_s: float
    segment_count: int
    burst_segment_count: int
    max_prob_burst: float
    mean_prob_burst: float
    max_segment_peak_db: float
    route_decision: str


def load_envelope_model(model_path: Path, device: str = "cpu"):
    torch, _ = ensure_torch()
    resolved = model_path.expanduser().resolve()
    checkpoint = torch.load(resolved, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint payload in {resolved}")

    model_kind = str(checkpoint.get("model_kind", ""))
    if not model_kind:
        raise ValueError(f"Checkpoint {resolved} does not define model_kind")

    model = build_model_from_kind(model_kind)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint {resolved} does not contain model_state_dict")
    model.load_state_dict(state_dict)
    model.eval()
    return model_kind, model


def predict_envelope(model: Any, envelope_db: np.ndarray) -> tuple[str, float, float]:
    torch, _ = ensure_torch()
    x = envelope_db.astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    tensor = torch.from_numpy(x[None, None, :])
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
    predicted_index = int(np.argmax(probabilities))
    return (
        INDEX_TO_LABEL[predicted_index],
        float(probabilities[LABEL_TO_INDEX["no_burst"]]),
        float(probabilities[LABEL_TO_INDEX["burst"]]),
    )


def analyze_file_with_envelope_model(
    file_path: Path,
    model: Any,
    channel: str = "ns",
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    segment_duration: float = DEFAULT_SEGMENT_DURATION,
    time_resolution: float = DEFAULT_TIME_RESOLUTION,
    frequency_resolution: float = DEFAULT_FREQUENCY_RESOLUTION,
    freq_min: float = DEFAULT_FREQ_MIN,
    freq_max: float = DEFAULT_FREQ_MAX,
    burst_threshold: float = DEFAULT_BURST_THRESHOLD,
) -> tuple[list[SegmentPrediction], FilePredictionSummary]:
    record_size, sample_count, data_offset = read_metadata(file_path)
    samples = load_samples(file_path, sample_count, record_size, data_offset)
    channel_index = CHANNEL_NAME_TO_INDEX[channel]
    channel_samples = extract_channel(
        samples=samples,
        sample_count=sample_count,
        record_size=record_size,
        channels=channels,
        channel_index=channel_index,
    )
    nperseg, hop = resolve_stft_parameters(
        samples=channel_samples,
        sample_rate=sample_rate,
        nperseg=4096,
        max_frames=4000,
        time_resolution=time_resolution,
        frequency_resolution=frequency_resolution,
    )
    resolved_min, resolved_max = resolve_frequency_band(sample_rate, freq_min, freq_max)
    segments = iter_time_segments(
        samples=channel_samples,
        sample_rate=sample_rate,
        segment_duration=segment_duration,
        nperseg=nperseg,
    )
    file_time = parse_time_from_name(file_path)
    duration_s = channel_samples.size / sample_rate

    segment_rows: list[SegmentPrediction] = []
    burst_segment_count = 0
    burst_probabilities: list[float] = []
    peak_values: list[float] = []

    for segment_index, start, stop, start_s, stop_s in segments:
        segment_samples = channel_samples[start:stop]
        spec_db, _, freq_axis = compute_spectrogram(segment_samples, nperseg=nperseg, hop=hop)
        spec_db, _ = crop_frequency_band(
            spec_db=spec_db,
            freq_axis=freq_axis,
            sample_rate=sample_rate,
            freq_min=resolved_min,
            freq_max=resolved_max,
        )
        envelope_db = compute_envelope_db(spec_db).astype(np.float32)
        predicted_label, prob_no_burst, prob_burst = predict_envelope(model, envelope_db)
        if prob_burst >= burst_threshold:
            predicted_label = "burst"
            burst_segment_count += 1
        else:
            predicted_label = "no_burst"

        burst_probabilities.append(prob_burst)
        peak_db = float(np.max(envelope_db))
        peak_values.append(peak_db)
        segment_rows.append(
            SegmentPrediction(
                file_path=file_path,
                file_time=file_time,
                channel=channel,
                segment_index=segment_index,
                segment_start=float(start_s),
                segment_end=float(stop_s),
                predicted_label=predicted_label,
                prob_no_burst=prob_no_burst,
                prob_burst=prob_burst,
                peak_value_db=peak_db,
                mean_value_db=float(np.mean(envelope_db)),
            )
        )

    route_decision = "burst_detected" if burst_segment_count else "empty"
    summary = FilePredictionSummary(
        file_path=file_path,
        file_time=file_time,
        channel=channel,
        duration_s=float(duration_s),
        segment_count=len(segment_rows),
        burst_segment_count=burst_segment_count,
        max_prob_burst=max(burst_probabilities) if burst_probabilities else 0.0,
        mean_prob_burst=float(np.mean(burst_probabilities)) if burst_probabilities else 0.0,
        max_segment_peak_db=max(peak_values) if peak_values else float("nan"),
        route_decision=route_decision,
    )
    return segment_rows, summary


def segment_rows_to_dicts(rows: list[SegmentPrediction]) -> list[dict[str, object]]:
    return [
        {
            "file_name": row.file_path.name,
            "file_path": str(row.file_path),
            "file_time": row.file_time,
            "channel": row.channel,
            "segment_index": row.segment_index,
            "segment_start": f"{row.segment_start:.6f}",
            "segment_end": f"{row.segment_end:.6f}",
            "predicted_label": row.predicted_label,
            "prob_no_burst": f"{row.prob_no_burst:.6f}",
            "prob_burst": f"{row.prob_burst:.6f}",
            "peak_value_db": f"{row.peak_value_db:.6f}",
            "mean_value_db": f"{row.mean_value_db:.6f}",
        }
        for row in rows
    ]


def summary_to_dict(summary: FilePredictionSummary, final_path: Path | None = None) -> dict[str, object]:
    return {
        "file_name": summary.file_path.name,
        "file_path": str(summary.file_path),
        "file_time": summary.file_time,
        "channel": summary.channel,
        "duration_s": f"{summary.duration_s:.6f}",
        "segment_count": summary.segment_count,
        "burst_segment_count": summary.burst_segment_count,
        "max_prob_burst": f"{summary.max_prob_burst:.6f}",
        "mean_prob_burst": f"{summary.mean_prob_burst:.6f}",
        "max_segment_peak_db": "" if np.isnan(summary.max_segment_peak_db) else f"{summary.max_segment_peak_db:.6f}",
        "route_decision": summary.route_decision,
        "final_path": "" if final_path is None else str(final_path),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_json_line(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
