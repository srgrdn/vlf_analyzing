# Repository Guidelines

## Mission And Context

This repository is a small research/operations toolkit for VLF broadband data. It is not a Python package yet; it is a collection of top-level CLI scripts that work together.

The current production-like workflow is:

1. Minute binary files `Broadband_Data_*.bin` appear in a watched directory.
2. `monitor_raw_data.py` waits until a file stops growing.
3. The monitor runs `detect_broadband_bursts.py` for both `ns` and `we`.
4. Empty files are deleted or moved away.
5. Non-empty files are routed to `processing1/`, and optionally copied to `processing2/`.
6. CSV reports are written under `reports/`.
7. `analyze_sferics_reports.py` builds summary statistics and plots from those reports.

When making changes, preserve this workflow unless the task explicitly asks to redesign it.

## Project Map

Main scripts:

- `plot_broadband_spectrogram.py`: common binary parsing helpers plus spectrogram rendering.
- `plot_broadband_oscillogram.py`: oscillogram CLI built on shared binary-reading helpers.
- `detect_broadband_bursts.py`: one-file detector; prints machine-readable summaries for automation.
- `monitor_raw_data.py`: long-running watcher/orchestrator for new minute files.
- `analyze_sferics_reports.py`: post-processing analytics over generated CSV reports.
- `run_whistler_classifier.py`: separate image-classification workflow for whistlers.

Generated artifacts:

- `reports/sferics_per_channel.csv`: one row per file/channel.
- `reports/sferics_summary.csv`: one row per minute file.
- `reports/sferics_events.csv`: one row per detected event.
- `reports/analysis/*`: derived plots and summary files from `analyze_sferics_reports.py`.

Operational directories outside or near `watch-dir`:

- `.monitor/`: monitor state and log.
- `empty/`: files with zero detections when not deleting them immediately.
- `processing1/`: primary routed non-empty raw files.
- `processing2/`: optional second routing target for another pipeline.
- `failed/`: files that could not be processed successfully.

## Critical Contracts

These stdout contracts are used by automation. Do not break them without updating the monitor in the same change:

- `TOTAL_DETECTIONS=...`
- `DETECTION_SUMMARY_JSON=...`
- `DETECTION_EVENTS_JSON=...`

Important behavioral contracts:

- `monitor_raw_data.py` must only process files matching `Broadband_Data_*.bin` by default.
- `Broadband_Sets_*.bin` must be ignored unless explicitly requested otherwise.
- The monitor must treat a file as ready only after its size/mtime are stable for the configured number of scan cycles.
- The detector must remain usable both interactively (`--show`) and in headless automation (`--no-save --allow-no-output`).
- Current working detection band is `20-30 kHz` unless the user changes CLI arguments.

## Safe Editing Rules

- Do not modify, delete, truncate, or overwrite real raw `.bin` data unless explicitly asked.
- Treat `reports/*.csv`, `reports/analysis/*`, `empty/*`, `processing1/*`, `processing2/*`, and `failed/*` as generated artifacts. They may be regenerated, but do not silently remove them unless the task requires it.
- Do not change filename parsing for `Broadband_Data_YYYY.MM.DD_HH.MM.SS.bin` lightly. Multiple reports depend on this timestamp format.
- Keep `monitor_raw_data.py` conservative with disk usage and CPU usage. Avoid introducing unnecessary duplicate copies, aggressive polling, or parallel detector launches by default.
- Preserve Windows-friendly behavior. This repo is actively used on Windows 7-era systems, so avoid Linux-only assumptions in operational scripts.

## Development Style

- Use Python 3, standard library first, `pathlib.Path` for paths, and 4-space indentation.
- Keep computational logic separate from CLI glue when possible.
- Prefer adding small pure functions over expanding `main()` further.
- When adding report columns, update both the writing code and the documentation in `README.md`.
- When changing monitor routing behavior, update comments/help text and README examples in the same patch.

## Validation Commands

Run these after relevant changes:

- `python3 -m py_compile *.py`
- `python3 detect_broadband_bursts.py --help`
- `python3 monitor_raw_data.py --help`
- `python3 analyze_sferics_reports.py --help`

Useful smoke checks:

- Headless detector run:
  - `python3 detect_broadband_bursts.py <file>.bin --channel ns --freq-min 20000 --freq-max 30000 --segment-duration 60 --time-resolution 0.002 --frequency-resolution 50 --threshold-mad 16 --no-save --allow-no-output`
- Analytics over existing reports:
  - `python3 analyze_sferics_reports.py --reports-dir reports`

If you change report formats or event accounting, validate against real or copied report files when possible.

## Reporting And Statistics Guidance

The current analysis questions this repo supports include:

- how many sferics were found per minute file;
- how detections split between `ns` and `we`;
- which channel dominates more often;
- what absolute times individual events occurred;
- peak level distributions per channel.

If you add new statistics or plots:

- prefer saving them under `reports/analysis/`;
- keep filenames stable and descriptive;
- keep the outputs useful without requiring notebook work.

## Windows Operational Notes

- Scripts are often run on Windows from `cmd`, with raw data located in a different directory than the scripts.
- Always assume paths may contain spaces and Cyrillic characters; examples and code should handle quoted paths correctly.
- Avoid relying on symlinks, daemons, or Unix-only schedulers unless explicitly asked.
- `#!/usr/bin/env python3` may remain in scripts; it is harmless on Windows and useful elsewhere.

## Agent Workflow Recommendations

When working as an agent in this repo:

1. Read `README.md` first if the task touches behavior or operator workflow.
2. Check whether the change affects detector output contracts, report schema, or monitor routing.
3. If changing a contract, update all dependent scripts in the same patch.
4. Prefer reversible operational changes over destructive ones.
5. Summarize any behavior changes in terms of:
   - detector output;
   - monitor routing;
   - report schema;
   - operator commands.

## Commit Guidance

Use short imperative commit subjects, for example:

- `Add per-event timing to sferics reports`
- `Make processing2 routing optional in monitor`
- `Add report analytics plots`

If a change affects operator workflow, mention that clearly in the commit body or PR description.
