# Threshold Pseudo-Labeling

This method uses `detect_broadband_bursts.py` as a weak labeler.

The detector creates labels automatically:

```text
burst     if at least one threshold peak is found in a 2 s segment
no_burst  if no threshold peaks are found
```

The neural network then learns from this pseudo-labeled dataset.

## Prepare Dataset

Run from the project root:

```powershell
python .\methods\threshold_pseudo_labeling\prepare_threshold_dataset.py .\raw_data --channel ns --freq-min 20000 --freq-max 30000 --time-resolution 0.002 --frequency-resolution 50 --segment-duration 2 --threshold-mad 8 --overwrite
```

Outputs:

```text
methods/threshold_pseudo_labeling/dataset/samples/
methods/threshold_pseudo_labeling/dataset/review/burst/
methods/threshold_pseudo_labeling/dataset/review/no_burst/
methods/threshold_pseudo_labeling/dataset/verify/burst/
methods/threshold_pseudo_labeling/dataset/verify/no_burst/
methods/threshold_pseudo_labeling/dataset/metadata.csv
```

## Train

Open:

```text
methods/threshold_pseudo_labeling/train_threshold_pseudo_labeling.ipynb
```

Select the project venv kernel and run cells from top to bottom.

Outputs:

```text
methods/threshold_pseudo_labeling/models/threshold_pseudo_envelope_1d.pt
methods/threshold_pseudo_labeling/reports/threshold_pseudo_envelope_1d_metrics.json
methods/threshold_pseudo_labeling/reports/threshold_pseudo_envelope_1d_predictions.csv
```

## Verify

Put manually checked PNGs into:

```text
methods/threshold_pseudo_labeling/dataset/verify/burst/
methods/threshold_pseudo_labeling/dataset/verify/no_burst/
```

Keep the original `pseudo_000123.png` filename. The notebook maps it back to the corresponding `.npz`.
