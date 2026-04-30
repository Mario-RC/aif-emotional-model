"""Aggregate annotator-filled spreadsheets into metrics and plots.

Replaces ``human_annotations_results.ipynb``. Run from this folder:

    python analyze_results.py

Prints Rank@1 / Rank@2 / mean / stdev / Task-2 emotion-hit tables per
annotator plus overall aggregates, saves bar-chart PDFs to
``hist/models/`` and ``hist/emotions/``, and finishes with Krippendorff
alpha per task on the shared IAA rows.
"""

from __future__ import annotations

import argparse
import logging

from src.config import PRINT_LABELS, TASKS
from src.metrics import (
    aggregate_emotion_hits, compute_emotion_hits, compute_iaa_alpha,
    compute_mean_std, compute_rank_at_k, iaa_uids_for,
)
from src.plots import (
    plot_hits_by_emotion, plot_hits_by_model, save_path_emotions,
    save_path_models,
)
from src.results_loader import load_all_annotators


logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("analyze_results")


# Shared IAA dialogue prefixes from the sampling protocol.
IAA_BASE_UIDS = (
    "SFTANNO-000006",
    "SFTANNO-000011",
    "SFTANNO-000018",
    "SFTANNO-000019",
    "SFTANNO-000025",
    "SFTANNO-000031",
    "SFTANNO-000066",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip rendering bar charts.",
    )
    args = parser.parse_args(argv)

    log.info("Loading annotators")
    results = load_all_annotators()
    log.info("Loaded %d annotators: %s",
             len(results), [r.annotator.name for r in results])

    # --- Rank@K and Mean/Std per annotator --------------------------------
    for res in results:
        for k in (1, 2):
            print(compute_rank_at_k(res, k=k).render())
            print()
        print(compute_mean_std(res).render())
        print()

    # --- Task 2 emotion hits ---------------------------------------------
    all_hits = [compute_emotion_hits(res) for res in results]
    overall = aggregate_emotion_hits(all_hits)

    total = sum(len(r.tasks[2]) for r in results)
    per = int(total / len(results)) if results else 0
    for hits in all_hits:
        print(f"{hits.annotator_name}  - Task 2")
        for label, n in zip(PRINT_LABELS, hits.total_hit_counts()):
            print(f"{label:<8} - {n} / {per}")
        print()

    print("OVERALL Task 2 - model hits")
    for label, n in zip(PRINT_LABELS, overall.sums):
        print(f"{label:<8} - {n} / {total}")

    if not args.no_plots:
        log.info("Rendering bar charts")
        plot_hits_by_model(
            overall.means, stds=overall.stds,
            ylim=per or 45, save_path=save_path_models(),
        )
        plot_hits_by_emotion(
            overall.per_emotion_mean,
            stds_by_emotion=overall.per_emotion_std,
            ylim=per or 45, save_path=save_path_emotions(),
        )

    # --- Inter-annotator agreement ---------------------------------------
    iaa_uids_by_task = {
        t.num: iaa_uids_for(t.num, IAA_BASE_UIDS) for t in TASKS
    }
    alpha_by_task = compute_iaa_alpha(results, iaa_uids_by_task)
    print("\nIAA")
    for num, value in alpha_by_task.items():
        print(f"TASK {num} {value:.3f}")


if __name__ == "__main__":
    main()
