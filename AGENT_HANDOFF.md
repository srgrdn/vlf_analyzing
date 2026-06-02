# Agent Handoff

## What This Branch Is

This worktree is for:

- `feature/2d-sferics-localization`

This branch is intentionally separated from:

- envelope-model production pipeline work
- old monitoring/reporting work from other branches

Stay focused on localization dataset + review + future 2D training.

## What Exists Right Now

Implemented:

- `build_localization_dataset.py`
- `build_localization_review_subset.py`
- `init_localization_corrections.py`
- `apply_localization_corrections.py`
- `summarize_localization_review.py`

Documented in:

- `README.md`
- `AGENT_STATUS.md`
- `AGENTS.md`

## Important Real-World State

The user has already run these on real data:

- `build_localization_dataset.py`
- `build_localization_review_subset.py`

Real outputs currently reported by the user:

- `dataset_localization/metadata.csv` with `6000` rows
- `dataset_localization/review_subset/manifest.csv` with `160` rows

This means the next human-meaningful step is probably not another infra script, but:

- reviewing and correcting `corrections.csv`
- using `python3 summarize_localization_review.py dataset_localization` to track progress

## What To Check First

When resuming:

1. `git status --short --branch`
2. `AGENT_STATUS.md`
3. `dataset_localization/review_subset/manifest.csv`
4. `dataset_localization/review_subset/corrections.csv` if it already exists

## Expected Next Development Options

### Option A: Manual Review Support

Good if the user wants to keep improving labels before training.

Possible next code tasks:

- helper to export only `manual_verified` + `manual_corrected`
- helper to mark train/val/test splits by `source_file`

### Option B: Training Skeleton

Good only after a first reviewed subset exists.

Likely next file:

- `methods/2d_cnn_localization/train_sferics_localization_2d.py`

Expected model shape:

- input: `spec_db`
- output: temporal heatmap
- post-processing: peak picking for event times/count

## Known Caveats

- Current labels are still mostly `threshold_auto`
- `we` is likely noisier and more artifact-prone than `ns`
- Do not present the current dataset as gold until reviewed corrections have been applied

## Handoff Discipline

Before ending your turn:

1. update `AGENT_STATUS.md`
2. if your change affects the next person materially, update this file too
3. keep commits milestone-sized and descriptive
