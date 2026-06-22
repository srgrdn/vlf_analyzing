# Full Result: 2D CNN inference over raw_vlf_data

Input: `/home/s_grudinin/code/university/vlf_analyzing/raw_vlf_data`

Model: `methods/2d_cnn_localization/models/sferics_localization_2d.pt`

Parameters:

- segment duration: `2.0` s
- frequency band: `20000-30000` Hz
- peak threshold: `0.35`
- min peak distance: `0.03` s

Summary:

- input files: `220`
- processed segments: `13200`
- total predicted events: `50140`
- ns events: `29237`
- we events: `20903`

Main files:

- `predictions/full_raw_2d_cnn_predictions.csv`
- `full_raw_2d_cnn_summary.json`
- `events_by_file_channel.csv`
- `events_by_hour_channel.csv`
- `events_by_time_of_day_channel.csv`
- `plots/`
