# Model Methods

The dataset is shared and stays in the project-level `dataset/` directory.
Each method has its own notebook, training script, models, and reports.

## 1D CNN Envelope

Path: `methods/1d_cnn_envelope/`

Input feature: `envelope_db` from `dataset/samples/*.npz`.

Use when we want a fast CPU baseline that sees only band-power dynamics over time.

Outputs:

```text
methods/1d_cnn_envelope/models/envelope_1d.pt
methods/1d_cnn_envelope/reports/envelope_1d_metrics.json
methods/1d_cnn_envelope/reports/envelope_1d_predictions.csv
```

## 2D CNN Spectrogram

Path: `methods/2d_cnn_spectrogram/`

Input feature: `spec_db` from `dataset/samples/*.npz`.

Use when we want the model to see the full time-frequency structure of a segment.

Outputs:

```text
methods/2d_cnn_spectrogram/models/spectrogram_2d.pt
methods/2d_cnn_spectrogram/reports/spectrogram_2d_metrics.json
methods/2d_cnn_spectrogram/reports/spectrogram_2d_predictions.csv
```

## 2D CNN Localization

Path: `methods/2d_cnn_localization/`

Input feature: `spec_db` from `dataset_localization/samples/*.npz`.

Target: `target_heatmap` from the same `.npz`.

Use when we want to predict a temporal event heatmap from the full spectrogram, then later post-process that heatmap into event times and event count.

Outputs:

```text
methods/2d_cnn_localization/models/sferics_localization_2d.pt
methods/2d_cnn_localization/reports/sferics_localization_2d_metrics.json
methods/2d_cnn_localization/reports/sferics_localization_2d_test_predictions.csv
```

The training notebook also includes a raw-file inference section for
`new_test_raw_data/Broadband_Data_2026.06.02_13.51.00.bin`. It reloads the
saved checkpoint, builds 2-second `20-30 kHz` spectrogram windows for `ns` and
`we`, writes predicted event times to `reports/`, and plots the most active
segments.

For repeatable non-notebook inference, use:

```bash
python3 methods/2d_cnn_localization/run_sferics_localization_2d.py \
  /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data \
  --channel both \
  --save-plots
```

For human verification of model predictions on new raw data, use:

```bash
python3 methods/2d_cnn_localization/verify_sferics_localization_2d_interactive.py \
  /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data \
  --channel both

python3 methods/2d_cnn_localization/summarize_sferics_localization_verification.py \
  --save-plots
```

## Threshold Pseudo-Labeling

Path: `methods/threshold_pseudo_labeling/`

Input labels: automatic labels produced by the threshold detector.

Use when we want to scale dataset creation without manually reviewing every segment.

Prepare the pseudo-labeled dataset first:

```powershell
python .\methods\threshold_pseudo_labeling\prepare_threshold_dataset.py .\raw_data --channel ns --freq-min 20000 --freq-max 30000 --time-resolution 0.002 --frequency-resolution 50 --segment-duration 2 --threshold-mad 8 --overwrite
```

Outputs:

```text
methods/threshold_pseudo_labeling/models/threshold_pseudo_envelope_1d.pt
methods/threshold_pseudo_labeling/reports/threshold_pseudo_envelope_1d_metrics.json
methods/threshold_pseudo_labeling/reports/threshold_pseudo_envelope_1d_predictions.csv
```

## Recommended Workflow

1. Clean labels in `dataset/review/burst` and `dataset/review/no_burst`.
2. Run `python sync_dataset_labels.py` from the project root.
3. Train `methods/1d_cnn_envelope/train_envelope_1d_cnn.ipynb`.
4. Train `methods/2d_cnn_spectrogram/train_spectrogram_2d_cnn.ipynb`.
5. Prepare and train `methods/threshold_pseudo_labeling/train_threshold_pseudo_labeling.ipynb`.
6. Compare the `reports/*_metrics.json` files.
