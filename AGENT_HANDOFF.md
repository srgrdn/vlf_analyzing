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
- `build_localization_active_review_subset.py`
- `init_localization_corrections.py`
- `apply_localization_corrections.py`
- `summarize_localization_review.py`
- `review_localization_interactive.py`
- `split_localization_reviewed.py`
- `methods/2d_cnn_localization/train_sferics_localization_2d.py`
- `methods/2d_cnn_localization/train_sferics_localization_2d.ipynb`
- `methods/2d_cnn_localization/run_sferics_localization_2d.py`
- `methods/2d_cnn_localization/verify_sferics_localization_2d_interactive.py`
- `methods/2d_cnn_localization/summarize_sferics_localization_verification.py`
- `methods/2d_cnn_localization/METHOD_DESCRIPTION_RU.md`
- `docs/BASIC_SIGNAL_PROCESSING_RU.md`
- `docs/THRESHOLD_DETECTOR_RU.md`
- `docs/TRAINING_DATASET_PREPARATION_RU.md`
- `docs/2D_CNN_EVENT_DETECTION_RU.md`
- `docs/METRICS_AND_QUALITY_EVALUATION_RU.md`
- `docs/MODEL_EVALUATION_RESULTS_RU.md`

Documented in:

- `README.md`
- `AGENT_STATUS.md`
- `AGENTS.md`
- `methods/2d_cnn_localization/METHOD_DESCRIPTION_RU.md`
- `docs/BASIC_SIGNAL_PROCESSING_RU.md`
- `docs/THRESHOLD_DETECTOR_RU.md`
- `docs/TRAINING_DATASET_PREPARATION_RU.md`
- `docs/2D_CNN_EVENT_DETECTION_RU.md`
- `docs/METRICS_AND_QUALITY_EVALUATION_RU.md`
- `docs/MODEL_EVALUATION_RESULTS_RU.md`

Documentation assets:

- `docs/assets/dataset_preparation/`
- `docs/assets/2d_cnn_detection/`
- `docs/assets/metrics_quality/`
- `docs/assets/model_evaluation_results/`

Generated diagnostic artifacts:

- `reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/`
- `full_result/` - full 2D CNN inference/statistics over all `raw_vlf_data` files (`220` files, `13200` segments, `50140` predicted events)
- `docs/dataset_reproducibility/dataset_localization_snapshot_2026-06-22/` - lightweight metadata snapshot kept before deleting the heavy generated dataset
- `docs/dataset_reproducibility/raw_data_snapshot_2026-06-22/` - lightweight manifest kept before deleting root-level raw input data

## Important Real-World State

The user has already run these on real data:

- `build_localization_dataset.py`
- `build_localization_review_subset.py`

Important cleanup state:

- the heavy generated `dataset_localization/` directory was deleted on 2026-06-22 to free disk space
- before deletion, lightweight metadata/corrections were copied to `docs/dataset_reproducibility/dataset_localization_snapshot_2026-06-22/`
- the deleted dataset had `6000` rows, `100` source raw files, `3000 ns` segments, `3000 we` segments
- final label sources before deletion: `351 manual_corrected`, `209 manual_verified`, `5440 threshold_auto`
- final reviewed split before deletion: `415 train`, `70 val`, `75 test`
- preserved files include final `metadata.csv`, pre-review/pre-split metadata backups, first review subset manifest/corrections, and v2 manifest/corrections
- scripts and documentation remain available so the dataset can be regenerated from new raw data
- root-level raw input directories were also deleted on 2026-06-22 to free disk space:
  - `raw_vlf_data/`
  - `raw_data/`
  - `new_test_raw_data/`
- before deletion, a `323`-file manifest was saved under `docs/dataset_reproducibility/raw_data_snapshot_2026-06-22/`
- restore compatible raw files under the same directory names before rerunning raw waveform workflows

This means the old review subset milestone was complete, but the generated dataset artifacts are no longer present. The next human-meaningful step after loading new raw data is:

- rebuild `dataset_localization` from the new raw data
- create/review/apply corrections using the existing review workflow
- regenerate reviewed train/val/test split
- train or rerun `methods/2d_cnn_localization/train_sferics_localization_2d.ipynb` only after the new reviewed dataset exists

## What To Check First

When resuming:

1. `git status --short --branch`
2. `AGENT_STATUS.md`
3. whether `dataset_localization/` has been regenerated
4. `docs/dataset_reproducibility/dataset_localization_snapshot_2026-06-22/README.md` for old counts and reproduction commands

## Expected Next Development Options

### Option A: Manual Review Support

Mostly complete for the first subset.

Possible next code tasks:

- helper to export only `manual_verified` + `manual_corrected`
- manual review of `dataset_localization/review_subset_v2`

### Option B: Training Skeleton

Implemented as a first baseline skeleton.

Likely next file:

- `methods/2d_cnn_localization/train_sferics_localization_2d.py`

Expected model shape:

- input: `spec_db`
- output: temporal heatmap
- post-processing: peak picking for event times/count

Current caveat:

- the active `.venv` used during implementation did not have `torch`, so only `py_compile`, `--help`, and metadata-selection checks were run

### Option C: Raw Inference

Implemented as a standalone CLI:

```bash
python3 methods/2d_cnn_localization/run_sferics_localization_2d.py \
  /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data \
  --channel both \
  --save-plots \
  --plot-top 6
```

Outputs:

- `methods/2d_cnn_localization/reports/raw_inference/sferics_localization_2d_raw_predictions.csv`
- `methods/2d_cnn_localization/reports/raw_inference/sferics_localization_2d_raw_summary.json`
- optional PNGs under `methods/2d_cnn_localization/reports/raw_inference/plots/`

This requires the same PyTorch environment used for notebook training.

### Option D: Raw Model Verification

Implemented as a human-in-the-loop verification workflow:

```bash
python3 methods/2d_cnn_localization/verify_sferics_localization_2d_interactive.py \
  /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data \
  --channel both
```

Then summarize:

```bash
python3 methods/2d_cnn_localization/summarize_sferics_localization_verification.py \
  --save-plots
```

Outputs:

- `methods/2d_cnn_localization/reports/raw_verification/sferics_localization_2d_raw_verification.csv`
- `methods/2d_cnn_localization/reports/raw_verification/sferics_localization_2d_verification_summary.json`
- `methods/2d_cnn_localization/reports/raw_verification/sferics_localization_2d_verification_segment_metrics.csv`
- optional plots under `methods/2d_cnn_localization/reports/raw_verification/plots/`

### Option E: Experimental Azimuth Estimation

Implemented as a separate experimental module, without changing the 2D CNN training/inference code:

```bash
python3 methods/azimuth_estimation/estimate_sferic_azimuth.py \
  --events-csv full_result/predictions/full_raw_2d_cnn_predictions.csv \
  --output-dir azimuth_results \
  --example-count 12
```

Outputs:

- `azimuth_results/sferic_azimuth_estimates.csv`
- `azimuth_results/azimuth_summary.json`
- `azimuth_results/AZIMUTH_ESTIMATION_REPORT.md`
- `azimuth_results/azimuth_histogram.png`
- `azimuth_results/azimuth_rose.png`
- `azimuth_results/ratio_we_ns_histogram.png`
- `azimuth_results/pca_linearity_histogram.png`
- `azimuth_results/azimuth_by_file_time.png`
- `azimuth_results/azimuth_direction_map.png`
- diagnostic examples under `azimuth_results/examples/`

Current full-run result:

- `50140` CNN detections loaded
- `37240` merged physical-event candidates
- `2825` reliable (`good`) azimuth estimates
- dominant reliable arrival-axis bin: `60-70 deg`
- mean reliable arrival axis: about `63.8 deg`

Important interpretation:

- this is not full source localization and does not estimate distance to the lightning discharge
- one station with two orthogonal loop antennas gives only a line of possible arrival with `180 deg` ambiguity
- candidates are saved as `azimuth_candidate_1` and `azimuth_candidate_2`
- calibration arguments exist but still use defaults: `gain_ns=1.0`, `gain_we=1.0`, `phase_shift_samples=0`, `antenna_angle_offset_deg=0`
- the direction map defaults to an Irkutsk receiver position: `lat=52.2864`, `lon=104.2807`
- future work should compare against external lightning-location data such as Blitzortung/WWLLN and calibrate channel gain/phase/orientation

## Known Caveats

- Current labels are still mostly `threshold_auto`
- `we` is likely noisier and more artifact-prone than `ns`
- Do not present the current dataset as gold until reviewed corrections have been applied
- The azimuth module is experimental: channel amplitude/phase calibration is not yet known, and the output is only a bearing line, not source coordinates.

## Handoff Discipline

Before ending your turn:

1. update `AGENT_STATUS.md`
2. if your change affects the next person materially, update this file too
3. keep commits milestone-sized and descriptive
