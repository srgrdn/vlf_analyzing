# Raw Data Snapshot Before Cleanup

This directory keeps a lightweight manifest of raw input files that were present before raw-data cleanup.

Cleanup date: 2026-06-22

## Preserved Files

- `raw_data_manifest.csv` - file-level manifest with:
  - raw root directory
  - relative path
  - absolute path before cleanup
  - size in bytes
  - modification time

## Removed Raw Directories

The following raw-data directories were removed from `/home/s_grudinin/code/university/vlf_analyzing` to free disk space:

- `raw_vlf_data/`
- `raw_data/`
- `new_test_raw_data/`

## Counts Before Cleanup

| Raw root | Files | Approx size |
|---|---:|---:|
| `raw_vlf_data` | `233` | `5.1G` |
| `raw_data` | `18` | `43M` |
| `new_test_raw_data` | `72` | `273M` |
| **Total** | `323` | `5.4G` |

## Reproduction Notes

To reproduce the previous workflows, restore compatible raw Broadband files under the same root-level directory names:

```text
/home/s_grudinin/code/university/vlf_analyzing/raw_vlf_data/
/home/s_grudinin/code/university/vlf_analyzing/raw_data/
/home/s_grudinin/code/university/vlf_analyzing/new_test_raw_data/
```

Important previous inputs:

- `raw_vlf_data/` was used for full 2D CNN inference statistics and aggregate azimuth estimation.
- `new_test_raw_data/Broadband_Data_2026.06.02_13.51.00.bin` was used for basic signal-processing plots, single-file 2D CNN statistics, and single-file azimuth experiments.
- `raw_vlf_data/june/` was used for June full-spectrogram examples.

Generated outputs and reports remain in the repository/worktree, but any rerun that needs raw waveforms requires restoring raw files first.

## Related Reproduction Docs

- `../dataset_localization_snapshot_2026-06-22/README.md` documents the deleted generated localization dataset.
- `../../BASIC_SIGNAL_PROCESSING_RU.md` describes the raw signal reading and plotting workflow.
- `../../2D_CNN_EVENT_DETECTION_RU.md` describes model inference on raw files.
