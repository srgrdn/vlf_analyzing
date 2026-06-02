# Agent Status

## Branch

- Branch: `feature/2d-sferics-localization`
- Base: `main`
- Scope: `spec_db -> 2D localization -> temporal heatmap -> event times + event count`

## Current State

Completed:

1. Localization dataset builder
2. Stratified review subset builder
3. Manual correction workflow
4. Interactive review workflow
5. First reviewed subset
6. Reviewed train/val/test split policy

Not started yet:

1. `2D CNN` training code
2. Inference code for the future localization model
3. Quantitative evaluation against reviewed labels

## Current Data Snapshot

From the latest user run:

- `dataset_localization/metadata.csv`
- `6000` segments total
- `3000` channel `ns`
- `3000` channel `we`
- `100` minute files

Review subset:

- `dataset_localization/review_subset/manifest.csv`
- `160` selected rows
- `80` for `ns`
- `80` for `we`

Observed subset distribution:

- `ns`: `20 empty`, `20 single`, `20 few`, `20 dense`
- `we`: `8 empty`, `27 single`, `25 few`, `20 dense`

Interpretation:

- `we` remains more active under current auto-labeling
- there were not enough `empty` examples in `we`, so subset selection used `coverage_fill`

## Current Branch Commits

- `7119151` `Add localization dataset builder for 2D sferics`
- `4d06e28` `Add localization review subset workflow`
- `c738f18` `Add localization correction workflow`

## Working Commands

Build localization dataset:

```bash
python3 build_localization_dataset.py /home/s_grudinin/code/university/vlf_analyzing/raw_vlf_data/raw_data \
  --channel both \
  --recursive \
  --freq-min 20000 \
  --freq-max 30000 \
  --segment-duration 2 \
  --time-resolution 0.002 \
  --frequency-resolution 50 \
  --overwrite
```

Build review subset:

```bash
python3 build_localization_review_subset.py dataset_localization
```

Create corrections template:

```bash
python3 init_localization_corrections.py dataset_localization/review_subset
```

Apply corrections:

```bash
python3 apply_localization_corrections.py dataset_localization
```

## Recommended Next Step

Best next milestone:

1. Use the existing `review_subset`
2. Fill `dataset_localization/review_subset/corrections.csv`
3. Apply corrections
4. Confirm first reviewed gold subset exists
5. Only then start `train_sferics_localization_2d.py`

If the user wants code work before manual review, the next acceptable engineering task is:

- implement the training skeleton for the future 2D localization model, but **do not pretend labels are already gold**

## Status Update Protocol

After each completed step, update this file.

Add a new item under `Progress Log` with:

- date
- step name
- files changed
- validation run
- outcome
- next recommended step

## Progress Log

### 2026-06-02 - Dataset Builder

- Step: implemented localization dataset builder
- Files: `build_localization_dataset.py`, `README.md`
- Validation:
  - `python3 -m py_compile ...`
  - `python3 build_localization_dataset.py --help`
  - smoke run on one minute file with `--channel both`
- Outcome:
  - `30` segments per channel confirmed on one minute file
  - metadata, `.npz`, and review PNG generation verified
- Next:
  - add review subset workflow

### 2026-06-02 - Review Subset Workflow

- Step: implemented stratified review subset builder
- Files: `build_localization_review_subset.py`, `README.md`
- Validation:
  - `python3 -m py_compile ...`
  - `python3 build_localization_review_subset.py --help`
  - smoke subset generation from a one-file localization dataset
- Outcome:
  - subset PNGs and `manifest.csv` produced
  - channel/bucket grouping verified
- Next:
  - add manual correction workflow

### 2026-06-02 - Manual Correction Workflow

- Step: implemented corrections template and apply pipeline
- Files: `init_localization_corrections.py`, `apply_localization_corrections.py`, `README.md`
- Validation:
  - `python3 -m py_compile ...`
  - `python3 init_localization_corrections.py --help`
  - `python3 apply_localization_corrections.py --help`
  - smoke cycle with one `manual_verified` and one `manual_corrected` sample
- Outcome:
  - metadata and `.npz` update together
  - backup metadata file created
  - PNGs copied into `review/verified` and `review/corrected`
- Next:
  - review real subset and create first gold labels

### 2026-06-02 - Agent Continuity Files

- Step: added multi-agent continuity and handoff files
- Files: `AGENTS.md`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`, `PROMPT_NEXT_AGENT.md`
- Validation:
  - reviewed file contents locally
  - confirmed branch status and current milestone context
- Outcome:
  - future agents now have a fixed resume order
  - status tracking is explicit and mandatory
  - a ready-to-paste prompt exists for the next agent
- Next:
  - use `corrections.csv` on the real `review_subset` and produce the first reviewed gold subset

### 2026-06-02 - Review Progress Summary CLI

- Step: implemented read-only review progress summary for localization corrections
- Files: `summarize_localization_review.py`, `README.md`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`
- Validation:
  - `python3 -m py_compile summarize_localization_review.py init_localization_corrections.py apply_localization_corrections.py`
  - `python3 summarize_localization_review.py --help`
  - `python3 summarize_localization_review.py dataset_localization`
- Outcome:
  - current real subset reports `160` total rows, `0` reviewed, `160` pending
  - CLI prints status/channel/bucket/quality summaries and next pending PNG paths
- Next:
  - fill `dataset_localization/review_subset/corrections.csv`, rerun the summary, then apply corrections when reviewed rows exist

### 2026-06-02 - Interactive Localization Reviewer

- Step: implemented matplotlib-based interactive review workflow
- Files: `review_localization_interactive.py`, `README.md`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`
- Validation:
  - `python3 -m py_compile review_localization_interactive.py summarize_localization_review.py init_localization_corrections.py apply_localization_corrections.py`
  - `python3 review_localization_interactive.py --help`
- Outcome:
  - reviewer shows `.npz` spectrograms with time/frequency axes
  - mouse clicks become corrected event times
  - hotkeys save `manual_verified`, `manual_corrected`, `artifact_suspected`, or `uncertain` rows to `corrections.csv` after each sample
- Next:
  - run `python3 review_localization_interactive.py dataset_localization`, review a small batch, then validate with summary and `apply_localization_corrections.py --dry-run`

### 2026-06-02 - Interactive Reviewer Click Retention Fix

- Step: fixed interactive reviewer redraw behavior so mouse clicks remain selected until saved
- Files: `review_localization_interactive.py`, `README.md`, `AGENT_STATUS.md`
- Validation:
  - `python3 -m py_compile review_localization_interactive.py summarize_localization_review.py init_localization_corrections.py apply_localization_corrections.py`
  - inspected current `corrections.csv` and found existing `manual_corrected` rows still match auto-labels from the earlier buggy run
- Outcome:
  - click selections are no longer reloaded away during redraw
  - `--review-status manual_corrected` can reopen only the affected rows
- Next:
  - rerun `python3 review_localization_interactive.py dataset_localization --review-status manual_corrected` and re-save corrected examples with the fixed reviewer

### 2026-06-02 - First Reviewed Localization Subset Applied

- Step: applied manual corrections for the real localization review subset
- Files: `dataset_localization/metadata.csv`, `dataset_localization/samples/*.npz`, `dataset_localization/review/verified/*.png`, `dataset_localization/review/corrected/*.png`, `dataset_localization/review_subset/corrections.csv`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`
- Validation:
  - `python3 apply_localization_corrections.py dataset_localization`
  - `python3 summarize_localization_review.py dataset_localization`
  - `python3 -m py_compile review_localization_interactive.py summarize_localization_review.py init_localization_corrections.py apply_localization_corrections.py`
  - metadata sanity check: `5840 threshold_auto`, `85 manual_corrected`, `75 manual_verified`
- Outcome:
  - `160/160` review subset rows are complete
  - reviewed rows are balanced by channel: `80 ns`, `80 we`
  - reviewed rows have `quality_flag=clean`
  - `160` reviewed PNGs are materialized under `review/verified` and `review/corrected`
- Next:
  - commit the review tooling/status milestone, then implement a train/val/test split policy for reviewed localization samples by `source_file`

### 2026-06-02 - Reviewed Split Policy

- Step: implemented source-file grouped train/val/test split assignment for reviewed localization rows
- Files: `split_localization_reviewed.py`, `README.md`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`
- Validation:
  - `python3 -m py_compile split_localization_reviewed.py review_localization_interactive.py summarize_localization_review.py init_localization_corrections.py apply_localization_corrections.py`
  - `python3 split_localization_reviewed.py --help`
  - `python3 split_localization_reviewed.py dataset_localization --dry-run`
  - `python3 split_localization_reviewed.py dataset_localization`
- Outcome:
  - reviewed rows only are assigned `train`/`val`/`test`
  - `threshold_auto` rows keep blank `split`
  - all reviewed rows from the same `source_file` stay in the same split
- Next:
  - start `2D CNN -> temporal heatmap` training skeleton using only `manual_verified` and `manual_corrected` rows with non-empty split
