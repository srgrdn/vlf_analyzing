# dataset_localization Snapshot Before Cleanup

This directory keeps the lightweight metadata needed to understand and reproduce the deleted `dataset_localization` artifacts.

Cleanup date: 2026-06-22

## Preserved Files

- `metadata.csv` - final metadata before cleanup
- `metadata.pre_review_backup.csv` - metadata backup before applying review corrections
- `metadata.pre_split_backup.csv` - metadata backup before reviewed train/val/test split
- `review_subset_manifest.csv` - first review subset manifest
- `review_subset_corrections.csv` - first review subset manual corrections
- `review_subset_v2_manifest.csv` - second active-review subset manifest
- `review_subset_v2_corrections.csv` - second active-review subset manual corrections

## Final Dataset Counts Before Cleanup

- Total rows: `6000`
- Source raw files: `100`
- Channels:
  - `ns`: `3000` segments
  - `we`: `3000` segments
- Label sources:
  - `manual_corrected`: `351`
  - `manual_verified`: `209`
  - `threshold_auto`: `5440`
- Splits:
  - `train`: `415`
  - `val`: `70`
  - `test`: `75`
  - blank: `5440`
- Quality flags:
  - `clean`: `560`
  - `unchecked`: `5440`

## Heavy Artifacts Removed

The following generated dataset artifacts can be recreated and were removed to save disk space:

- `dataset_localization/samples/` - `.npz` training samples
- `dataset_localization/review/` - generated review PNGs
- `dataset_localization/review_subset/` - review-subset PNGs plus copied CSVs
- `dataset_localization/review_subset_v2/` - active-review PNGs plus copied CSVs
- remaining `dataset_localization/` files after the metadata snapshot was copied

Approximate size before cleanup: `7.1G`.

## Reproduction Commands

From the worktree root:

```bash
python3 build_localization_dataset.py --help
python3 build_localization_review_subset.py --help
python3 init_localization_corrections.py --help
python3 apply_localization_corrections.py --help
python3 build_localization_active_review_subset.py --help
python3 review_localization_interactive.py --help
python3 split_localization_reviewed.py --help
```

Typical workflow for new raw data:

```bash
python3 build_localization_dataset.py <raw_data_dir> dataset_localization
python3 build_localization_review_subset.py dataset_localization --overwrite
python3 init_localization_corrections.py dataset_localization/review_subset
python3 review_localization_interactive.py dataset_localization
python3 apply_localization_corrections.py dataset_localization
python3 build_localization_active_review_subset.py dataset_localization --overwrite
python3 init_localization_corrections.py dataset_localization/review_subset_v2
python3 review_localization_interactive.py dataset_localization --review-subset-dir dataset_localization/review_subset_v2
python3 apply_localization_corrections.py dataset_localization --corrections dataset_localization/review_subset_v2/corrections.csv
python3 split_localization_reviewed.py dataset_localization
```

Notes:

- The exact command-line arguments may need to be adjusted for the new raw-data directory and desired subset sizes.
- The preserved correction files document the old manual-review decisions, but they should not be blindly applied to a newly generated dataset unless `sample_id` and source segments are intentionally identical.
- The 2D CNN training/inference code is not part of the deleted dataset and remains under `methods/2d_cnn_localization/`.
