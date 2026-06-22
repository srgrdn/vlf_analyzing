#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively review localization samples with matplotlib. "
            "Mouse clicks mark event times inside the segment; keyboard shortcuts "
            "write review_status values to corrections.csv after each accepted sample."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to dataset_localization.",
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        default=None,
        help="Path to corrections.csv. Default: <dataset_dir>/review_subset/corrections.csv",
    )
    parser.add_argument(
        "--review-subset-dir",
        type=Path,
        default=None,
        help="Path to review subset directory. Default: <dataset_dir>/review_subset, or parent of --corrections.",
    )
    parser.add_argument(
        "--start-sample-id",
        default=None,
        help="Start from this sample_id if it is present in corrections.csv.",
    )
    parser.add_argument("--channel", choices=("ns", "we"), default=None, help="Review only one channel.")
    parser.add_argument(
        "--bucket",
        choices=("empty", "single", "few", "dense"),
        default=None,
        help="Review only one subset bucket.",
    )
    parser.add_argument(
        "--include-reviewed",
        action="store_true",
        help="Include rows that already have review_status instead of showing only pending rows.",
    )
    parser.add_argument(
        "--review-status",
        default=None,
        help=(
            "Review only rows with this review_status. Use 'pending' for blank status. "
            "This implies --include-reviewed for non-pending statuses."
        ),
    )
    parser.add_argument("--cmap", default="jet", help="Spectrogram colormap. Default: jet")
    parser.add_argument("--db-low", type=float, default=2.0, help="Lower dB display percentile. Default: 2")
    parser.add_argument("--db-high", type=float, default=99.8, help="Upper dB display percentile. Default: 99.8")
    return parser.parse_args()


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return rows, list(reader.fieldnames)


def write_csv_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean(value: str | None) -> str:
    return (value or "").strip()


def parse_times(raw: str | None) -> list[float]:
    text = clean(raw)
    if not text:
        return []
    values = json.loads(text)
    if not isinstance(values, list):
        raise ValueError("event times must be a JSON list")
    return sorted(float(value) for value in values)


def format_times(times: list[float]) -> str:
    return json.dumps([round(float(time_s), 6) for time_s in sorted(times)])


def build_review_order(
    correction_rows: list[dict[str, str]],
    manifest_by_id: dict[str, dict[str, str]],
    include_reviewed: bool,
    review_status: str | None,
    channel: str | None,
    bucket: str | None,
) -> list[int]:
    order: list[int] = []
    for index, row in enumerate(correction_rows):
        sample_id = row["sample_id"]
        manifest_row = manifest_by_id[sample_id]
        current_status = clean(row.get("review_status")) or "pending"
        if review_status is not None and current_status != review_status:
            continue
        if review_status is None and not include_reviewed and current_status != "pending":
            continue
        if channel is not None and manifest_row["channel"] != channel:
            continue
        if bucket is not None and manifest_row["bucket"] != bucket:
            continue
        order.append(index)
    return order


class InteractiveReviewer:
    def __init__(
        self,
        *,
        dataset_dir: Path,
        corrections_path: Path,
        correction_rows: list[dict[str, str]],
        fieldnames: list[str],
        manifest_by_id: dict[str, dict[str, str]],
        order: list[int],
        cmap: str,
        db_low: float,
        db_high: float,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.corrections_path = corrections_path
        self.correction_rows = correction_rows
        self.fieldnames = fieldnames
        self.manifest_by_id = manifest_by_id
        self.order = order
        self.cmap = cmap
        self.db_low = db_low
        self.db_high = db_high
        self.position = 0
        self.clicked_times: list[float] = []
        self.auto_times: list[float] = []
        self.time_min = 0.0
        self.time_max = 0.0
        self.current_spec_db: np.ndarray | None = None
        self.current_freq_axis: np.ndarray | None = None
        self.fig = None
        self.ax = None

    def run(self) -> None:
        import matplotlib.pyplot as plt

        if not self.order:
            print("No matching samples to review.")
            return

        self.fig, self.ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.show_current()
        plt.show()

    def current_row(self) -> dict[str, str]:
        return self.correction_rows[self.order[self.position]]

    def current_manifest_row(self) -> dict[str, str]:
        return self.manifest_by_id[self.current_row()["sample_id"]]

    def load_current_times(self, npz: np.lib.npyio.NpzFile) -> None:
        row = self.current_row()
        self.auto_times = [float(value) for value in np.asarray(npz["event_times_s"], dtype=np.float32)]
        if clean(row.get("corrected_event_times_s_json")):
            self.clicked_times = parse_times(row["corrected_event_times_s_json"])
        else:
            self.clicked_times = list(self.auto_times)

    def load_current_sample(self) -> None:
        row = self.current_row()
        manifest_row = self.current_manifest_row()
        npz_path = self.dataset_dir / manifest_row["npz_path"]
        npz = np.load(npz_path, allow_pickle=True)
        time_axis = np.asarray(npz["time_axis"], dtype=np.float32)
        self.current_spec_db = np.asarray(npz["spec_db"], dtype=np.float32)
        self.current_freq_axis = np.asarray(npz["freq_axis"], dtype=np.float32)
        self.time_min = float(time_axis[0])
        self.time_max = float(time_axis[-1])
        self.load_current_times(npz)

    def redraw_current(self) -> None:
        assert self.ax is not None
        row = self.current_row()
        manifest_row = self.current_manifest_row()
        sample_id = row["sample_id"]
        if self.current_spec_db is None or self.current_freq_axis is None:
            raise RuntimeError("No current sample is loaded.")

        vmin, vmax = np.percentile(self.current_spec_db, [self.db_low, self.db_high])
        self.ax.clear()
        self.ax.imshow(
            self.current_spec_db,
            origin="lower",
            aspect="auto",
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            extent=(
                self.time_min,
                self.time_max,
                float(self.current_freq_axis[0]),
                float(self.current_freq_axis[-1]),
            ),
            interpolation="nearest",
        )
        for event_time in self.auto_times:
            self.ax.axvline(event_time, color="white", linestyle="--", linewidth=1.0, alpha=0.9)
        for event_time in self.clicked_times:
            self.ax.axvline(event_time, color="red", linestyle="-", linewidth=1.3, alpha=0.95)

        status = clean(row.get("review_status")) or "pending"
        title = (
            f"{self.position + 1}/{len(self.order)} {sample_id} | "
            f"{manifest_row['channel']}/{manifest_row['bucket']} | "
            f"auto_events={manifest_row['event_count']} | status={status}\n"
            "Click events. Keys: v=verified, enter/c=save corrected, "
            "backspace=undo, r=reset auto, x=clear, a=artifact, u=uncertain, s=skip, q=quit"
        )
        self.ax.set_title(title, fontsize=10)
        self.ax.set_xlabel("Time inside segment, s")
        self.ax.set_ylabel("Frequency, Hz")
        self.ax.figure.canvas.draw_idle()
        print(
            f"[{self.position + 1}/{len(self.order)}] {sample_id} "
            f"{manifest_row['channel']}/{manifest_row['bucket']} "
            f"auto={self.auto_times} selected={self.clicked_times}"
        )

    def show_current(self) -> None:
        self.load_current_sample()
        self.redraw_current()

    def on_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None:
            return
        event_time = min(max(float(event.xdata), self.time_min), self.time_max)
        self.clicked_times.append(event_time)
        self.clicked_times.sort()
        self.redraw_current()

    def on_key(self, event) -> None:
        key = event.key or ""
        if key == "q":
            print("Stopping review.")
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
            self.redraw_current()
            return
        if key == "r":
            self.clicked_times = list(self.auto_times)
            self.redraw_current()
            return
        if key == "x":
            self.clicked_times = []
            self.redraw_current()
            return
        if key == "v":
            self.save_current("manual_verified", corrected_times=None, quality_flag="")
            self.advance()
            return
        if key in ("enter", "c"):
            self.save_current("manual_corrected", corrected_times=self.clicked_times, quality_flag="")
            self.advance()
            return
        if key == "a":
            self.save_current("artifact_suspected", corrected_times=None, quality_flag="artifact_suspected")
            self.advance()
            return
        if key == "u":
            self.save_current("uncertain", corrected_times=None, quality_flag="uncertain")
            self.advance()
            return

    def save_current(self, review_status: str, corrected_times: list[float] | None, quality_flag: str) -> None:
        row = self.current_row()
        row["review_status"] = review_status
        row["quality_flag"] = quality_flag
        if corrected_times is None:
            row["corrected_event_times_s_json"] = ""
        else:
            row["corrected_event_times_s_json"] = format_times(corrected_times)
        write_csv_rows(self.corrections_path, self.correction_rows, self.fieldnames)
        print(f"Saved {row['sample_id']}: {review_status}")

    def advance(self) -> None:
        self.position += 1
        if self.position >= len(self.order):
            print("Review queue complete.")
            if self.fig is not None:
                import matplotlib.pyplot as plt

                plt.close(self.fig)
            return
        self.show_current()


def validate_manifest_links(manifest_rows: list[dict[str, str]], correction_rows: list[dict[str, str]]) -> None:
    manifest_ids = {row["sample_id"] for row in manifest_rows}
    correction_ids = {row["sample_id"] for row in correction_rows}
    unknown_ids = sorted(correction_ids - manifest_ids)
    missing_ids = sorted(manifest_ids - correction_ids)
    if unknown_ids:
        raise ValueError(f"Corrections contain sample_id values missing from manifest.csv: {', '.join(unknown_ids[:10])}")
    if missing_ids:
        raise ValueError(f"Corrections are missing manifest sample_id values: {', '.join(missing_ids[:10])}")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    corrections_path = (
        args.corrections.expanduser().resolve()
        if args.corrections
        else None
    )
    review_subset_dir = (
        args.review_subset_dir.expanduser().resolve()
        if args.review_subset_dir
        else (corrections_path.parent if corrections_path else dataset_dir / "review_subset")
    )
    corrections_path = corrections_path or review_subset_dir / "corrections.csv"
    manifest_path = review_subset_dir / "manifest.csv"

    manifest_rows, _ = read_csv_rows(manifest_path)
    correction_rows, fieldnames = read_csv_rows(corrections_path)
    validate_manifest_links(manifest_rows, correction_rows)
    manifest_by_id = {row["sample_id"]: row for row in manifest_rows}
    order = build_review_order(
        correction_rows=correction_rows,
        manifest_by_id=manifest_by_id,
        include_reviewed=args.include_reviewed or (args.review_status not in (None, "pending")),
        review_status=args.review_status,
        channel=args.channel,
        bucket=args.bucket,
    )

    if args.start_sample_id:
        matching_positions = [
            position
            for position, row_index in enumerate(order)
            if correction_rows[row_index]["sample_id"] == args.start_sample_id
        ]
        if not matching_positions:
            raise ValueError(f"start sample_id is not in the current review queue: {args.start_sample_id}")
        order = order[matching_positions[0] :]

    reviewer = InteractiveReviewer(
        dataset_dir=dataset_dir,
        corrections_path=corrections_path,
        correction_rows=correction_rows,
        fieldnames=fieldnames,
        manifest_by_id=manifest_by_id,
        order=order,
        cmap=args.cmap,
        db_low=args.db_low,
        db_high=args.db_high,
    )
    reviewer.run()


if __name__ == "__main__":
    main()
