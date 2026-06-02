## Branch Scope

This worktree is dedicated to the `feature/2d-sferics-localization` branch.

The purpose of this branch is narrow and explicit:

- build a clean `spec_db -> 2D localization` pipeline;
- prepare localization-ready datasets and review workflows;
- avoid mixing in unrelated monitoring, production routing, or old model-pipeline code from other branches.

If you are continuing work here, stay inside this scope unless the user explicitly expands it.

## Required Historical Context

Before continuing work that depends on older project discussion, inspect:

- `docs/codex-history/FULL_CODEX_CONTEXT.md`

This is historical context only. Do not treat it as source code and do not edit it unless explicitly requested.

## Canonical Files For This Branch

Current branch-specific workflow is centered around:

- `build_localization_dataset.py`
- `build_localization_review_subset.py`
- `init_localization_corrections.py`
- `apply_localization_corrections.py`
- `README.md`
- `AGENT_STATUS.md`
- `AGENT_HANDOFF.md`
- `PROMPT_NEXT_AGENT.md`

Existing shared low-level helpers from `main` that this branch may reuse:

- `plot_broadband_spectrogram.py`
- `detect_broadband_bursts.py`

## Current Development Stage

This branch is **not** at training yet.

Current completed milestones:

1. Localization dataset builder
2. Stratified review subset builder
3. Manual correction workflow

Next expected milestone:

4. First baseline `2D CNN -> temporal heatmap` training pipeline

## What To Avoid

Do not add or reintroduce here unless the user explicitly asks:

- envelope-model production monitor scripts
- old reporting pipelines from other branches
- whistler-classifier workflow changes
- unrelated cleanup of `main`
- GUI review tools

Keep this branch focused on the localization research path.

## Mandatory Status Tracking

After **every completed development step**, update:

- `AGENT_STATUS.md`

At minimum, append or update:

- what was done
- which files changed
- validation that was run
- what remains next

If you are handing work to another agent, also update:

- `AGENT_HANDOFF.md`

## Required Workflow For Every Agent

Follow this order:

1. Read `AGENT_STATUS.md`
2. Read `AGENT_HANDOFF.md`
3. Read only the relevant part of `docs/codex-history/FULL_CODEX_CONTEXT.md` if older context is needed
4. Inspect current git status
5. Perform one clearly scoped milestone or sub-step
6. Run validation commands relevant to your change
7. Update `AGENT_STATUS.md`
8. Update `AGENT_HANDOFF.md` if the next agent would benefit

## Validation Expectations

For any code change, run only the validations relevant to that step.

Common checks for this branch:

```bash
python3 -m py_compile build_localization_dataset.py \
  build_localization_review_subset.py \
  init_localization_corrections.py \
  apply_localization_corrections.py \
  plot_broadband_spectrogram.py \
  detect_broadband_bursts.py
```

And, when relevant:

```bash
python3 build_localization_dataset.py --help
python3 build_localization_review_subset.py --help
python3 init_localization_corrections.py --help
python3 apply_localization_corrections.py --help
```

## Dataset And Review Conventions

Localization dataset root:

```text
dataset_localization/
  samples/
  review/
    auto/
    verified/
    corrected/
  review_subset/
  metadata.csv
```

Key metadata fields:

- `sample_id`
- `channel`
- `event_count`
- `event_times_s_json`
- `label_source`
- `quality_flag`

Review status values used by the correction workflow:

- `manual_verified`
- `manual_corrected`
- `artifact_suspected`
- `uncertain`

## Commit Discipline

Prefer small, milestone-shaped commits.

Current branch history already follows this pattern:

- dataset builder
- review subset
- correction workflow

Continue in the same style.

## If You Need A Starting Prompt

Use:

- `PROMPT_NEXT_AGENT.md`

It is intended to be pasted directly into another Codex/agent session.
