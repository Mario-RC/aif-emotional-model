"""Build the annotator-facing Excel files from the SFT test predictions.

Replaces ``human_annotations_generation.ipynb``. Run from this folder:

    python generate_annotations.py

Outputs:

* ``data/demonstration_data_emotional_balanced_test_results_dial.csv`` —
  wide DataFrame with every model's prediction side by side.
* ``data/task2_reference_labels.xlsx`` — long Task-2 reference-label
  table, with one row per prompt/model response.
* ``data/tasks/task{1..4}.xlsx`` — the 4 raw task DataFrames.
* ``data/aux/anno{i}_t{j}_aux.xlsx`` — per-annotator splits *with*
  internal columns (used later for metric computation).
* ``data/final/anno{i}_<slug>.xlsx`` — per-annotator spreadsheets in
  the schema delivered to the annotators.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import (
    ANNOTATORS, DATA_DIR, RESULTS_DIR, TASK2_REFERENCE_LABELS_PATH,
    WIDE_CSV_PATH,
)
from src.parsing import build_wide_dataframe
from src.tasks import (
    TASKS, build_task2_reference_labels, build_task_dataframe,
    save_aux_slices, save_final_slices, save_task2_reference_labels,
    save_task_dataframes,
    split_across_annotators,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("generate_annotations")


def _annotation_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c == "USER_EMOTION"
        or (
            isinstance(c, str)
            and c.startswith("MODEL")
            and ("QUALITY" in c or "EMOTION" in c)
        )
    ]


def _has_human_filled_results(results_dir=RESULTS_DIR) -> bool:
    """Detect submitted human annotations without modifying ``results/``."""
    for annotator in ANNOTATORS:
        if not annotator.included:
            continue
        for task in TASKS:
            path = (results_dir / annotator.name
                    / f"anno{annotator.index}_{task.slug}.xlsx")
            if not path.exists():
                continue
            df = pd.read_excel(path)
            for col in _annotation_columns(df):
                if df[col].notna().any():
                    return True
    return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wide-csv", type=Path, default=WIDE_CSV_PATH,
        help="Where to save the combined wide DataFrame.",
    )
    parser.add_argument(
        "--skip-save-wide", action="store_true",
        help="Skip writing the wide CSV (useful during development).",
    )
    args = parser.parse_args(argv)

    log.info("Loading SFT predictions and building wide DataFrame")
    wide = build_wide_dataframe()
    if not args.skip_save_wide:
        args.wide_csv.parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(args.wide_csv, index=False, encoding="utf-8")
        log.info("Wrote %s (%d rows)", args.wide_csv, len(wide))

    if _has_human_filled_results() and TASK2_REFERENCE_LABELS_PATH.exists():
        log.info(
            "Keeping existing %s because human-filled results are present",
            TASK2_REFERENCE_LABELS_PATH,
        )
    else:
        log.info("Building Task-2 reference-label table")
        task2_reference = build_task2_reference_labels(wide)
        save_task2_reference_labels(task2_reference)
        log.info(
            "Wrote %s (%d rows)",
            TASK2_REFERENCE_LABELS_PATH, len(task2_reference),
        )

    log.info("Building per-task DataFrames")
    task_dfs = {t.num: build_task_dataframe(wide, t) for t in TASKS}
    save_task_dataframes(task_dfs)

    log.info("Splitting across annotators")
    slices = split_across_annotators(task_dfs)
    for slice_ in slices:
        sizes = {n: len(df) for n, df in slice_.tasks.items()}
        log.info("  anno%d %s -> %s", slice_.annotator.index,
                 slice_.annotator.name, sizes)

    log.info("Writing auxiliary files")
    save_aux_slices(slices)

    log.info("Writing annotator-facing files")
    save_final_slices(slices)

    log.info("Done. Output under %s", DATA_DIR)


if __name__ == "__main__":
    main()
