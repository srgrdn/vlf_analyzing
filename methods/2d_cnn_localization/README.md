# 2D CNN Localization

Baseline training skeleton for reviewed sferics localization samples.

Full Russian method description:

```text
methods/2d_cnn_localization/METHOD_DESCRIPTION_RU.md
```

Input:

- `spec_db` from `dataset_localization/samples/*.npz`

Target:

- `target_heatmap` from the same `.npz`

Rows used:

- `label_source` in `manual_verified`, `manual_corrected`
- `quality_flag=clean`
- non-empty `split`

Recommended interactive run:

```bash
jupyter notebook methods/2d_cnn_localization/train_sferics_localization_2d.ipynb
```

The notebook saves the best validation checkpoint to:

```text
methods/2d_cnn_localization/models/sferics_localization_2d.pt
```

After training, use the final notebook sections to:

- reload the saved checkpoint without retraining;
- run inference on `new_test_raw_data/Broadband_Data_2026.06.02_13.51.00.bin`;
- inspect predicted event heatmaps and peak-picked event times for 2-second `20-30 kHz` segments.

CLI run:

```bash
python3 methods/2d_cnn_localization/train_sferics_localization_2d.py \
  --dataset-dir dataset_localization
```

Raw inference CLI:

```bash
python3 methods/2d_cnn_localization/run_sferics_localization_2d.py \
  /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data \
  --channel both
```

Save diagnostic spectrogram+heatmap plots for the most active segments:

```bash
python3 methods/2d_cnn_localization/run_sferics_localization_2d.py \
  /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data \
  --channel both \
  --save-plots \
  --plot-top 6
```

Show plots interactively instead of only saving CSV/JSON:

```bash
python3 methods/2d_cnn_localization/run_sferics_localization_2d.py \
  /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data/Broadband_Data_2026.06.02_13.51.00.bin \
  --channel both \
  --show \
  --plot-top 3
```

Interactive verification on new raw data:

```bash
python3 methods/2d_cnn_localization/verify_sferics_localization_2d_interactive.py \
  /home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data \
  --channel both
```

Hotkeys:

- click on the spectrogram/heatmap: add corrected event time;
- `v`: model prediction is correct (`model_verified`);
- `enter` or `c`: save clicked/red times as manual correction;
- `f`: model produced only false positives; save corrected events as empty;
- `m`: model missed events; save clicked/red times as corrected events;
- `backspace`: remove last click;
- `r`: reset corrected times to model prediction;
- `x`: clear corrected times;
- `a`: artifact segment;
- `u`: uncertain segment;
- `s`: skip;
- `q`: quit.

Summarize verification and save plots:

```bash
python3 methods/2d_cnn_localization/summarize_sferics_localization_verification.py \
  --save-plots
```

Outputs:

```text
methods/2d_cnn_localization/models/sferics_localization_2d.pt
methods/2d_cnn_localization/reports/sferics_localization_2d_metrics.json
methods/2d_cnn_localization/reports/sferics_localization_2d_test_predictions.csv
methods/2d_cnn_localization/reports/raw_inference/sferics_localization_2d_raw_predictions.csv
methods/2d_cnn_localization/reports/raw_inference/sferics_localization_2d_raw_summary.json
methods/2d_cnn_localization/reports/raw_verification/sferics_localization_2d_raw_verification.csv
methods/2d_cnn_localization/reports/raw_verification/sferics_localization_2d_verification_summary.json
```
