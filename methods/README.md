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
