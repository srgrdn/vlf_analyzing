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
7. First `2D CNN -> temporal heatmap` training skeleton
8. Active review subset v2 builder
9. Expanded reviewed subset v2 applied
10. Notebook checkpoint reload and raw-file inference workflow
11. Standalone raw-file inference CLI for the saved 2D localization model
12. Interactive raw-file model verification workflow
13. Full Russian method description document
14. Basic signal-processing description and diagnostic artifacts
15. Russian threshold detector description document
16. Russian training dataset preparation and labeling document
17. Russian 2D CNN event detection description document with diagrams and metric plots
18. Russian metrics and quality evaluation chapter with formulas and plots
19. Russian model evaluation results chapter with plots, histograms, and conclusions
20. Full 2D CNN inference statistics over all `raw_vlf_data`

Not started yet:

1. Choosing the final `peak_threshold` by validation metrics
2. Quantitative comparison of the `560`-row model against the first `160`-row smoke run
3. Running human verification on a new raw-file verification set

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

### 2026-06-02 - 2D CNN Localization Training Skeleton

- Step: implemented baseline PyTorch training workflow for `spec_db -> target_heatmap`
- Files: `methods/2d_cnn_localization/train_sferics_localization_2d.py`, `methods/2d_cnn_localization/train_sferics_localization_2d.ipynb`, `methods/2d_cnn_localization/README.md`, `methods/2d_cnn_localization/__init__.py`, `methods/README.md`, `README.md`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`
- Validation:
  - `python3 -m py_compile methods/2d_cnn_localization/train_sferics_localization_2d.py split_localization_reviewed.py review_localization_interactive.py summarize_localization_review.py init_localization_corrections.py apply_localization_corrections.py`
  - `python3 methods/2d_cnn_localization/train_sferics_localization_2d.py --help`
  - metadata selection sanity check: `160` reviewed rows, split distribution `112 train`, `25 val`, `23 test`
- Outcome:
  - notebook is the recommended interactive training entrypoint with split summary, sample preview, loss plots, prediction table, and worst-example plots
  - script uses only `manual_verified` and `manual_corrected` rows with `quality_flag=clean` and non-empty split
  - model predicts a temporal heatmap with the same length as `target_heatmap`
  - reports include validation/test heatmap MSE/MAE and simple peak-picking preview CSVs
  - user smoke-trained `5` epochs on CPU; val heatmap MSE reached about `0.0124`, test heatmap MSE about `0.0114`
  - added event-level precision/recall/F1 and event-count MAE reporting with configurable match tolerance
  - first event metrics at `peak_threshold=0.35`, `match_tolerance=0.04`: val F1 about `0.858`, test F1 about `0.800`
  - added notebook peak-threshold sweep for validation/test F1 and event-count MAE
- Next:
  - run the notebook threshold sweep, choose threshold by validation F1/count MAE, then optionally train for `15-20` epochs

### 2026-06-02 - Active Review Subset v2

- Step: implemented active-review subset builder for expanding reviewed localization labels after the first model
- Files: `build_localization_active_review_subset.py`, `summarize_localization_review.py`, `review_localization_interactive.py`, `README.md`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`
- Validation:
  - `python3 -m py_compile build_localization_active_review_subset.py summarize_localization_review.py review_localization_interactive.py init_localization_corrections.py apply_localization_corrections.py`
  - `python3 build_localization_active_review_subset.py --help`
  - `python3 build_localization_active_review_subset.py dataset_localization --overwrite`
  - `python3 init_localization_corrections.py dataset_localization/review_subset_v2`
  - `python3 summarize_localization_review.py dataset_localization --review-subset-dir dataset_localization/review_subset_v2`
- Outcome:
  - `review_subset_v2` can be built from unreviewed rows only
  - selection prioritizes model-error source files when prediction CSVs exist, plus `we`, `few`, `dense`, and negative coverage
  - summary and interactive reviewer now support arbitrary review subset dirs
  - generated `dataset_localization/review_subset_v2` with `400` pending rows: `200 ns`, `200 we`; buckets: `35 empty`, `167 single`, `118 few`, `80 dense`
  - sanity check confirmed `0` previously reviewed rows in v2 and `400` `threshold_auto` rows
- Next:
  - manually review `dataset_localization/review_subset_v2` interactively, apply corrections with `--corrections`, rerun split, then retrain notebook

### 2026-06-02 - Expanded Reviewed Subset v2 Applied

- Step: applied second-pass active-review corrections and regenerated reviewed train/val/test split
- Files: `dataset_localization/metadata.csv`, `dataset_localization/samples/*.npz`, `dataset_localization/review/verified/*.png`, `dataset_localization/review/corrected/*.png`, `dataset_localization/review_subset_v2/corrections.csv`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`
- Validation:
  - `python3 apply_localization_corrections.py dataset_localization --corrections dataset_localization/review_subset_v2/corrections.csv --dry-run`
  - `python3 apply_localization_corrections.py dataset_localization --corrections dataset_localization/review_subset_v2/corrections.csv`
  - `python3 split_localization_reviewed.py dataset_localization`
  - `python3 summarize_localization_review.py dataset_localization --review-subset-dir dataset_localization/review_subset_v2 --limit 5`
  - metadata sanity check: `560` reviewed rows, `5440 threshold_auto`, no reviewed source_file split intersections
- Outcome:
  - `review_subset_v2` is `400/400` reviewed: `266 manual_corrected`, `134 manual_verified`
  - total reviewed dataset is now `560`: `351 manual_corrected`, `209 manual_verified`
  - reviewed rows are channel-balanced: `280 ns`, `280 we`
  - split distribution is `415 train`, `70 val`, `75 test`
  - `threshold_auto` rows keep blank split
- Next:
  - rerun `methods/2d_cnn_localization/train_sferics_localization_2d.ipynb` on the expanded split and compare event-level metrics against the first `160`-row smoke run

### 2026-06-03 - Notebook Raw-File Inference

- Step: added notebook sections for loading the saved 2D localization checkpoint and running it on a new raw Broadband file
- Files: `methods/2d_cnn_localization/train_sferics_localization_2d.ipynb`, `methods/2d_cnn_localization/README.md`, `methods/README.md`, `README.md`, `AGENT_STATUS.md`
- Validation:
  - notebook JSON load sanity check
  - checkpoint exists at `methods/2d_cnn_localization/models/sferics_localization_2d.pt`
  - raw file found at `/home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data/Broadband_Data_2026.06.02_13.51.00.bin`
  - `python3 -m py_compile methods/2d_cnn_localization/train_sferics_localization_2d.py plot_broadband_spectrogram.py`
- Outcome:
  - notebook can reload the saved model without retraining
  - raw inference builds 2-second `20-30 kHz` spectrogram windows for `ns` and `we`
  - predictions are written to `methods/2d_cnn_localization/reports/Broadband_Data_2026.06.02_13.51.00_2d_cnn_predictions.csv`
  - plotting cell overlays predicted event times on raw spectrograms and predicted heatmaps
- Next:
  - run the new notebook cells after training, inspect the most active raw-file segments, and choose `peak_threshold` by validation F1/count MAE before treating raw predictions as operational results

### 2026-06-03 - Raw Inference CLI

- Step: added a standalone CLI for running the saved 2D localization model on raw Broadband files
- Files: `methods/2d_cnn_localization/run_sferics_localization_2d.py`, `methods/2d_cnn_localization/README.md`, `methods/README.md`, `README.md`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`
- Validation:
  - `python3 -m py_compile methods/2d_cnn_localization/run_sferics_localization_2d.py methods/2d_cnn_localization/train_sferics_localization_2d.py plot_broadband_spectrogram.py`
  - `python3 methods/2d_cnn_localization/run_sferics_localization_2d.py --help`
- Outcome:
  - CLI loads `models/sferics_localization_2d.pt`
  - accepts a raw `.bin` file or a directory of `Broadband_Data_*.bin`
  - processes `ns`, `we`, or both channels
  - builds 2-second `20-30 kHz` spectrogram windows by default
  - writes raw prediction CSV and summary JSON under `methods/2d_cnn_localization/reports/raw_inference/`
  - optional `--save-plots` saves spectrogram+predicted-heatmap PNGs for top active segments
  - optional `--show` displays selected plots interactively
- Caveat:
  - full inference was not run from this shell because this shell does not have PyTorch installed; use the same `.venv`/Jupyter environment where training succeeded
- Next:
  - run CLI on `new_test_raw_data`, inspect saved top-segment plots for both channels, and compare CLI output with notebook raw inference

### 2026-06-03 - Interactive Raw Verification Workflow

- Step: added an interactive verification workflow for checking saved-model predictions on raw Broadband data
- Files: `methods/2d_cnn_localization/verify_sferics_localization_2d_interactive.py`, `methods/2d_cnn_localization/summarize_sferics_localization_verification.py`, `methods/2d_cnn_localization/README.md`, `methods/README.md`, `README.md`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`
- Validation:
  - `python3 -m py_compile methods/2d_cnn_localization/verify_sferics_localization_2d_interactive.py methods/2d_cnn_localization/summarize_sferics_localization_verification.py methods/2d_cnn_localization/run_sferics_localization_2d.py methods/2d_cnn_localization/train_sferics_localization_2d.py plot_broadband_spectrogram.py`
  - `python3 methods/2d_cnn_localization/verify_sferics_localization_2d_interactive.py --help`
  - `python3 methods/2d_cnn_localization/summarize_sferics_localization_verification.py --help`
- Outcome:
  - verifier builds model predictions on raw `Broadband_Data_*.bin` segments and opens a matplotlib review window
  - white lines show model event times; red lines show user-confirmed/corrected event times
  - hotkeys save `model_verified`, `manual_corrected`, `false_positive`, `missed_events`, `artifact_suspected`, or `uncertain`
  - verification CSV is persisted after each reviewed segment
  - summary CLI computes event-level precision/recall/F1, event-count MAE, status counts, channel metrics, and optional plots/histograms
- Caveat:
  - full interactive run requires the same PyTorch environment used for model training
- Next:
  - user should prepare a new raw verification set, run the interactive verifier, then summarize with `--save-plots`

### 2026-06-04 - Russian Method Description

- Step: added a full Russian description of the 2D CNN sferics localization method
- Files: `methods/2d_cnn_localization/METHOD_DESCRIPTION_RU.md`, `methods/2d_cnn_localization/README.md`, `AGENT_STATUS.md`
- Validation:
  - checked that the document exists and has `805` lines
  - checked README reference with `rg`
- Outcome:
  - document explains raw data assumptions, dataset construction, auto-labeling, manual review, active review v2, train/val/test split, target heatmap construction, normalization, 2D CNN architecture, training objective, peak-threshold post-processing, metrics, current results, notebook workflow, CLI inference, interactive verification, limitations, and next steps
- Next:
  - keep this document updated when the model architecture, dataset size, metrics, or verification protocol changes

### 2026-06-04 - Basic Signal Processing Documentation And Artifacts

- Step: documented the non-neural Broadband/VLF signal processing pipeline and generated diagnostic plots for the newest raw file
- Files: `docs/BASIC_SIGNAL_PROCESSING_RU.md`, `README.md`, `reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/README.md`, generated PNGs under `reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/`
- Validation:
  - `python3 -m py_compile plot_broadband_spectrogram.py plot_broadband_oscillogram.py`
  - generated full spectrograms for `ns` and `we`
  - generated full oscillograms for `ns` and `we`
  - generated 2-second `20-30 kHz` spectrogram segments for `ns` and `we`
  - generated 2-second oscillogram segments for `ns` and `we`
  - counted `124` PNG files total: `4` full plots and `120` segmented plots
- Outcome:
  - basic processing document explains binary parsing, channel extraction, dBFS oscillograms, STFT spectrograms, frequency-band cropping, segmentation, and the relation to the neural model
  - artifacts for `/home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data/Broadband_Data_2026.06.02_13.51.00.bin` are stored in `reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/`
- Next:
  - use these artifacts as a reference for comparing raw signal structure, spectrogram structure, and model heatmap predictions

### 2026-06-04 - Threshold Detector Description

- Step: added a full Russian description of the non-neural threshold detector
- Files: `docs/THRESHOLD_DETECTOR_RU.md`, `README.md`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`
- Validation:
  - checked that `docs/THRESHOLD_DETECTOR_RU.md` exists and has `731` lines
  - checked README reference with `rg`
  - `python3 -m py_compile detect_broadband_bursts.py build_localization_dataset.py`
- Outcome:
  - document explains detector purpose, STFT power computation, frequency-band cropping, band aggregation, dB envelope, smoothing, robust MAD threshold, `threshold-mad`, local peak picking, minimum peak distance, segmentation, standalone detector outputs, use in `build_localization_dataset.py`, target heatmap generation from threshold labels, relation to review workflow, relation to 2D CNN, threshold pseudo-labeling, parameters, strengths, limitations, and current role as weak labeler
- Next:
  - keep the detector document in sync if threshold defaults or dataset auto-labeling behavior changes

### 2026-06-08 - Training Dataset Preparation Description

- Step: added a full Russian description of training dataset preparation and labeling, with supporting figures
- Files: `docs/TRAINING_DATASET_PREPARATION_RU.md`, `docs/assets/dataset_preparation/*.png`, `README.md`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`
- Validation:
  - checked that `docs/TRAINING_DATASET_PREPARATION_RU.md` exists and has `779` lines
  - generated `4` PNG figures under `docs/assets/dataset_preparation/`
  - verified all markdown image links resolve
- Outcome:
  - document explains raw-to-dataset construction, `.npz` contents, `metadata.csv`, threshold auto-labeling, target heatmap construction, review subset v1, corrections template, interactive review, applying corrections, reviewed split policy, active review v2, final reviewed counts, training-row selection, review progress CLI, status semantics, relationship to CNN training, verification separation, reproducibility, limitations, and next steps
  - figures cover the dataset preparation pipeline, current dataset counts, review status lifecycle, and review subset distribution
- Next:
  - keep figures and counts updated when new review subsets or dataset versions are created

### 2026-06-08 - 2D CNN Event Detection Description

- Step: added a detailed Russian description of event detection with the trained 2D CNN localization model
- Files: `docs/2D_CNN_EVENT_DETECTION_RU.md`, `docs/assets/2d_cnn_detection/*.png`, `README.md`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`
- Validation:
  - generated `16` PNG figures under `docs/assets/2d_cnn_detection/`
  - verified figure files exist with `find docs/assets/2d_cnn_detection -maxdepth 1 -type f -name '*.png'`
  - checked source metrics from `methods/2d_cnn_localization/reports/sferics_localization_2d_metrics.json`
  - checked raw verification summary from `methods/2d_cnn_localization/reports/raw_verification/sferics_localization_2d_verification_summary.json`
- Outcome:
  - document explains CNN inference from raw data, target heatmap math, normalization, exact model architecture, MSE/MAE training objective, peak-picking post-processing, `peak_threshold`, event matching, val/test metrics, raw inference CLI, interactive verification workflow, current verification results, limitations, and next improvements
  - figures cover inference pipeline, architecture, target heatmap construction, example input/target, training curves, val/test metrics, TP/FP/FN counts, count-error histograms, heatmap-error histograms, raw inference channel summary, and raw verification summaries
- Next:
  - if a new model is trained or a new blind verification set is reviewed, regenerate the metric figures and update the numbers in `docs/2D_CNN_EVENT_DETECTION_RU.md`

### 2026-06-09 - Metrics And Quality Evaluation Chapter

- Step: added a Russian chapter describing metrics and quality evaluation for the threshold detector, 2D CNN model, and interactive verification workflow
- Files: `docs/METRICS_AND_QUALITY_EVALUATION_RU.md`, `docs/assets/metrics_quality/*.png`, `README.md`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`
- Validation:
  - generated `9` PNG figures under `docs/assets/metrics_quality/`
  - verified all markdown image links resolve
  - checked source CNN metrics from `methods/2d_cnn_localization/reports/sferics_localization_2d_metrics.json`
  - checked source raw verification metrics from `methods/2d_cnn_localization/reports/raw_verification/sferics_localization_2d_verification_summary.json`
- Outcome:
  - document explains signal-level, heatmap-level, event-level, count-level, and human-verification quality checks
  - includes formulas for MSE, MAE, event matching, TP/FP/FN, precision, recall, F1, event-count MAE, and threshold sweep selection
  - includes current val/test and raw-verification results with supporting plots
- Next:
  - when a threshold detector baseline is evaluated on the same raw verification set, add its numbers to this chapter for direct method comparison

### 2026-06-09 - Model Evaluation Results Chapter

- Step: added section `4.2. Результаты оценки моделей` with result-oriented plots, histograms, channel breakdowns, and conclusions
- Files: `docs/MODEL_EVALUATION_RESULTS_RU.md`, `docs/assets/model_evaluation_results/*.png`, `README.md`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`
- Validation:
  - generated `15` PNG assets under `docs/assets/model_evaluation_results/`
  - verified all `14` markdown image links resolve
  - computed channel-level val/test metrics from `sferics_localization_2d_val_predictions.csv` and `sferics_localization_2d_test_predictions.csv`
  - checked raw verification summary from `sferics_localization_2d_verification_summary.json`
- Outcome:
  - document summarizes train/val/test setup, learning curves, val/test event metrics, TP/FP/FN, channel-specific results, count-error histograms, test error analysis, raw verification results, and final conclusions
  - added `Таблица 1. Оценка качества работы модели 2D CNN локализации`
  - added separate event-detection error matrices for validation, test, and raw verification
  - added a standalone TP/FN/FP bar chart for the independent 2D CNN raw verification set
  - document explicitly notes that the threshold detector still needs a fair direct comparison against CNN on the same blind verification set
- Next:
  - run a threshold-detector baseline on the same raw verification segments and add a direct comparison table to this chapter

### 2026-06-17 - Full Raw Data 2D CNN Statistics

- Step: ran the saved 2D CNN localization model over all raw files under `/home/s_grudinin/code/university/vlf_analyzing/raw_vlf_data` and generated aggregate statistics
- Files: `full_result/README.md`, `full_result/REPORT_RU.md`, `full_result/full_raw_2d_cnn_summary.json`, `full_result/predictions/full_raw_2d_cnn_predictions.csv`, `full_result/events_by_file_channel.csv`, `full_result/events_by_hour_channel.csv`, `full_result/events_by_time_of_day_channel.csv`, `full_result/plots/*.png`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`
- Validation:
  - processed `220` raw `Broadband_Data_*.bin` files
  - processed `13200` two-second segments across `ns` and `we`
  - verified `full_result/REPORT_RU.md` has `9` image links and no missing plot files
  - listed generated files with `find full_result -maxdepth 2 -type f`
- Outcome:
  - total predicted events: `50140`
  - `ns`: `29237` events, `132.90` events/min, mean `4.43` events/segment
  - `we`: `20903` events, `95.01` events/min, mean `3.17` events/segment
  - all files in the processed set are after noon, so before-noon counts are `0`
  - generated plots for channel totals, per-minute intensity, events per segment, events over file time, intensity over file time, before/after noon, hourly intensity, and `max_heatmap`
- Next:
  - if morning raw files become available, rerun this workflow so before/after-noon comparison is meaningful

### 2026-06-21 - Experimental Azimuth Estimation From NS/WE Loops

- Step: added a separate experimental azimuth-estimation module for already detected sferic events using the two orthogonal loop-antenna channels `ns` and `we`
- Files: `methods/azimuth_estimation/estimate_sferic_azimuth.py`, `azimuth_results/sferic_azimuth_estimates.csv`, `azimuth_results/AZIMUTH_ESTIMATION_REPORT.md`, `azimuth_results/*.png`, `azimuth_results/examples/*.png`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`
- Validation:
  - `python3 -m py_compile methods/azimuth_estimation/estimate_sferic_azimuth.py`
  - `python3 methods/azimuth_estimation/estimate_sferic_azimuth.py --help`
  - smoke run with `--max-events 300 --example-count 4 --output-dir azimuth_results_smoke`
  - full run with `--example-count 12 --output-dir azimuth_results`
  - generated `azimuth_results/azimuth_direction_map.png` from the existing azimuth CSV and verified the file exists
  - verified expected CSV, report, summary JSON, six main plots, and twelve diagnostic example PNGs exist
- Outcome:
  - loaded `50140` CNN detections from `full_result/predictions/full_raw_2d_cnn_predictions.csv`
  - merged close `ns`/`we` detections into `37240` physical-event candidates using the default `0.04 s` tolerance
  - estimated RMS ratio and PCA polarization axis on paired raw windows for each event
  - saved `2825` events as `good` quality and kept unreliable events with quality flags instead of deleting them
  - dominant reliable arrival-axis bin is `60-70 deg`; mean reliable arrival axis is about `63.8 deg`
  - added a map-style bearing-line plot centered on Irkutsk with the dominant axis and opposite 180-degree candidate
  - report explicitly states that this is a single-station line-of-arrival estimate with `180 deg` ambiguity, not source coordinates or distance
- Next:
  - compare the estimated azimuth distribution with external lightning-location data such as Blitzortung/WWLLN, and calibrate `gain_ns`, `gain_we`, `phase_shift_samples`, and `antenna_angle_offset_deg` if reference events become available

### 2026-06-21 - Single June File Azimuth Estimate

- Step: ran the experimental azimuth estimator for the specific raw file `Broadband_Data_2026.06.02_13.51.00.bin`
- Files: `reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/azimuth_estimation/*`, `AGENT_STATUS.md`
- Validation:
  - input predictions: `reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/2d_cnn_statistics/raw_inference/sferics_localization_2d_raw_predictions.csv`
  - command: `python3 methods/azimuth_estimation/estimate_sferic_azimuth.py --events-csv reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/2d_cnn_statistics/raw_inference/sferics_localization_2d_raw_predictions.csv --output-dir reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/azimuth_estimation --example-count 8`
  - verified CSV, summary JSON, report, six plots, and four diagnostic examples were generated
- Outcome:
  - loaded `391` model detections and merged them into `296` physical-event candidates
  - strict reliable events: `4`
  - strict `good` peak bin: `50-60 deg`; strict mean arrival axis: about `63.1 deg`
  - events without `single_channel_detection`: `65`, with peak bin `60-70 deg` and mean arrival axis about `65.4 deg`
  - all valid events also peak at `60-70 deg`, but this includes many low-SNR/single-channel events and should be treated as lower confidence
- Next:
  - use this single-file result only as a preliminary bearing estimate; for stronger conclusions prefer the full `azimuth_results` aggregate or compare with external lightning-location data

### 2026-06-21 - Single June File Azimuth Without 90 Degree Rotation

- Step: reran the same single-file azimuth estimate with `--antenna-angle-offset-deg -90` to inspect the alternative interpretation where PCA magnetic-axis rotation is not applied
- Files: `reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/azimuth_estimation_no_90deg/*`, `AGENT_STATUS.md`
- Validation:
  - command: `python3 methods/azimuth_estimation/estimate_sferic_azimuth.py --events-csv reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/2d_cnn_statistics/raw_inference/sferics_localization_2d_raw_predictions.csv --output-dir reports/basic_signal_processing/Broadband_Data_2026.06.02_13.51.00/azimuth_estimation_no_90deg --antenna-angle-offset-deg -90 --example-count 8`
  - verified CSV, summary JSON, report, and plots were generated
- Outcome:
  - loaded `391` model detections and merged them into `296` physical-event candidates
  - strict reliable events: `4`
  - strict `good` peak bin shifted from `50-60 deg` to `140-150 deg`; strict mean shifted from about `63.1 deg` to about `153.1 deg`
  - events without `single_channel_detection` shifted from peak `60-70 deg` and mean `65.4 deg` to peak `150-160 deg` and mean `155.4 deg`
  - this confirms the alternative interpretation is the same bearing structure rotated by `90 deg`
- Next:
  - decide between the `+90 deg` and `no +90 deg` interpretations only after comparing with external lightning-direction data or a known calibration source

### 2026-06-22 - Resulting Dataset Slide Figure

- Step: generated a slide-ready figure summarizing the resulting `dataset_localization` dataset
- Files: `docs/assets/dataset_preparation/resulting_dataset_slide_summary.png`, `AGENT_STATUS.md`
- Validation:
  - read counts directly from `dataset_localization/metadata.csv`
  - visually checked the generated PNG for readable labels and non-overlapping layout
- Outcome:
  - figure shows `100` raw files for each channel `ns` and `we`
  - figure shows `3000` segments per channel, `6000` total segments
  - figure shows `560` manually reviewed segments and `5440` automatic/unchecked segments
  - figure shows reviewed split: `415 train`, `70 val`, `75 test`
- Next:
  - use this figure on the presentation slide titled `Получившийся набор данных`

### 2026-06-22 - Heavy Dataset Cleanup With Reproducibility Snapshot

- Step: removed the heavy generated `dataset_localization/` directory while preserving lightweight metadata needed to understand and reproduce the old dataset
- Files: `docs/dataset_reproducibility/dataset_localization_snapshot_2026-06-22/*`, `AGENT_STATUS.md`, `AGENT_HANDOFF.md`
- Validation:
  - copied final `metadata.csv`, metadata backups, review subset manifests, and correction CSVs before deletion
  - created `docs/dataset_reproducibility/dataset_localization_snapshot_2026-06-22/README.md` with counts and reproduction commands
  - verified `dataset_localization/` no longer exists
  - verified the worktree size dropped to about `289M`
- Outcome:
  - removed approximately `7.1G` of generated dataset artifacts
  - preserved final old dataset counts: `6000` rows, `100` source files, `3000 ns` segments, `3000 we` segments
  - preserved final old reviewed split: `415 train`, `70 val`, `75 test`
  - preserved final old label counts: `351 manual_corrected`, `209 manual_verified`, `5440 threshold_auto`
  - code, reports, docs, trained model artifacts, and raw data were not deleted
- Next:
  - when new raw data is available, regenerate `dataset_localization` using the documented commands and then redo review/split/training as needed
