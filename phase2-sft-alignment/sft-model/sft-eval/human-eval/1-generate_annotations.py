"""Build the annotator-facing Excel files from the SFT test predictions.

Replaces ``human_annotations_generation.ipynb``. Run from this folder:

    python generate_annotations.py

Outputs:

* ``data/demonstration_data_emotional_balanced_test_results_dial.csv`` —
  wide DataFrame with every model's prediction side by side.
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

from src.config import DATA_DIR, WIDE_CSV_PATH
from src.parsing import build_wide_dataframe
from src.tasks import (
    TASKS, build_task_dataframe, save_aux_slices, save_final_slices,
    save_task_dataframes, split_across_annotators,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("generate_annotations")


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
