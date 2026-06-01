# SFT Demonstration Dataset Manifest

This directory contains the scripts used to generate and format the paid SFT
demonstration dialogue source.

## Paid Raw Source

The raw paid generation artifacts are stored under `data/`:

- `sft_demonstration_dataset_checkpoint_0.csv` through `sft_demonstration_dataset_checkpoint_9.csv`
- `sft_demonstration_dataset.csv`

The checkpoint files were originally produced with legacy historical names
(`demonstation_data_checkpoint_ 0.csv` through `demonstation_data_checkpoint_ 9.csv`).
They have been renamed to match the current generator output naming convention.
The file contents were not changed.

The canonical raw source for the current pipeline is:

- `data/sft_demonstration_dataset.csv`

The checkpoint files are historical recovery/audit artifacts for the paid
generation run. They are not the canonical input for the formatting pipeline.

## Column Order

The raw CSV column order is fixed as:

`PROMPT`, `GENERATION_ID`, `COMPLETION`, `TOPIC`, `EMOTIONS`

Scripts should still read columns by name, but `1-generate_sft_demonstration_dataset.py`
writes this explicit order so future regenerations remain stable and comparable
with the historical paid CSVs.
