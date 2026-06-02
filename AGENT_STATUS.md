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

Not started yet:

1. Reviewing and correcting a real subset into a gold set
2. Train/val/test split policy implementation
3. `2D CNN` training code
4. Inference code for the future localization model
5. Quantitative evaluation against reviewed labels

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
