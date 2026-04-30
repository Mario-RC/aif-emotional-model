"""Aggregate human-annotation XLSX files and report Task 1 / Task 2 metrics.

Reads every annotator's ``task1.xlsx`` / ``task2.xlsx`` from
``results/<annotator>/`` and produces:

* ``Task 1``: per-system @1 / @2 hit rates (SFT / PPO / DPO).
* ``Task 2``: per-system EMPATHY / EMOTION / ENGAGEMENT adequacy counts.
* Bar plots: dimension overall + per-level breakdown for each adequacy column.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_ANNOTATORS = ["anmol", "david", "mario", "noel"]
RESPONSE_TYPES = ["SFT_TEST", "PPO_TEST", "DPO_TEST"]
RESPONSE_TYPE_LABEL = {"SFT_TEST": "SFT", "PPO_TEST": "PPO", "DPO_TEST": "DPO"}

EXPRESSION_LEVEL_TO_INT = {
    "Very High": 5,
    "High": 4,
    "Medium": 3,
    "Low": 2,
    "Very Low": 1,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_annotations(annotators: list[str], task: str) -> pd.DataFrame:
    """Concatenate every annotator's ``task<n>.xlsx`` (skipping missing ones)."""
    frames = [
        pd.read_excel(f"results/{annotator}/{task}.xlsx").drop(columns=["COMMENTS"])
        for annotator in annotators
        if os.path.exists(f"results/{annotator}/{task}.xlsx")
    ]
    return pd.concat(frames, ignore_index=True)


def attach_response_type(anno_t2: pd.DataFrame, data_test_df: pd.DataFrame) -> pd.DataFrame:
    """Look up each Task-2 RESPONSE in the SFT/PPO/DPO columns of ``data_test_df`` and tag it."""
    anno_t2 = anno_t2.copy()
    anno_t2["RESPONSE_R3"] = anno_t2["RESPONSE"].str.split("(NEUTRAL)").str[-1]
    anno_t2["RESPONSE_TYPE"] = ""

    for index, row in anno_t2.iterrows():
        response = row["RESPONSE_R3"]
        for col in RESPONSE_TYPES:
            for resp in data_test_df[col]:
                if response in resp:
                    anno_t2.at[index, "RESPONSE_TYPE"] = col
                    break
            if anno_t2.at[index, "RESPONSE_TYPE"]:
                break
        if not anno_t2.at[index, "RESPONSE_TYPE"]:
            anno_t2.at[index, "RESPONSE_TYPE"] = "NOT_FOUND"
    return anno_t2.drop(columns=["RESPONSE_R3"])


# ---------------------------------------------------------------------------
# Task 1 metrics
# ---------------------------------------------------------------------------

def calculate_rank_counts(df: pd.DataFrame, rank_values: list[int] = [1, 2]) -> tuple[dict, dict]:
    """For each cutoff in ``rank_values``, compute the @rank hit rate per system + std."""
    counts: dict[str, float] = {}
    stds: dict[str, float] = {}
    column_to_system = {"RESPONSE_1_RANK": "sft", "RESPONSE_2_RANK": "ppo", "RESPONSE_3_RANK": "dpo"}

    for rank in rank_values:
        for col, system in column_to_system.items():
            counts[f"{system}_at_{rank}"] = sum(df[col] <= rank) / len(df)
            stds[f"{system}_at_{rank}"] = df[col].std()
    return counts, stds


def report_task1(anno_t1: pd.DataFrame) -> None:
    counts, stds = calculate_rank_counts(anno_t1)
    print("ANNOS @1 - Task 1")
    for system in ("sft", "ppo", "dpo"):
        print(f"{system.upper()} - {counts[f'{system}_at_1']:.3f} ± {stds[f'{system}_at_1']:.3f}")
    print("\nANNOS @2 - Task 1")
    for system in ("sft", "ppo", "dpo"):
        print(f"{system.upper()} - {counts[f'{system}_at_2']:.3f} ± {stds[f'{system}_at_2']:.3f}")


# ---------------------------------------------------------------------------
# Task 2 metrics
# ---------------------------------------------------------------------------

def _map_expression_level(expression_list: list[str]) -> list[int]:
    return [
        EXPRESSION_LEVEL_TO_INT[next(k for k in EXPRESSION_LEVEL_TO_INT if k in level)]
        for level in expression_list
    ]


def normalize_task2(anno_t2: pd.DataFrame) -> pd.DataFrame:
    """Convert EXPRESSION_LEVEL strings to int triples and Agree/Disagree to 1/0."""
    df = anno_t2.copy()
    df["EXPRESSION_LEVEL"] = df["EXPRESSION_LEVEL"].str.split("\n").apply(_map_expression_level)
    adequacy_cols = ["EMPATHY_ADEQUACY", "EMOTION_ADEQUACY", "ENGAGEMENT_ADEQUACY"]
    df[adequacy_cols] = df[adequacy_cols].replace({"Agree": 1, "Disagree": 0})
    df[["EMPATHY_LEVEL", "EMOTION_LEVEL", "ENGAGEMENT_LEVEL"]] = pd.DataFrame(
        df["EXPRESSION_LEVEL"].tolist(), index=df.index
    )
    return df


def _adequacy_counts_per_system(anno_t2: pd.DataFrame) -> dict[str, list[int]]:
    counts: dict[str, list[int]] = {}
    for response_type in RESPONSE_TYPES:
        sub = anno_t2[anno_t2["RESPONSE_TYPE"] == response_type]
        counts[response_type] = [
            sub["EMPATHY_ADEQUACY"].value_counts().get(1, 0),
            sub["EMOTION_ADEQUACY"].value_counts().get(1, 0),
            sub["ENGAGEMENT_ADEQUACY"].value_counts().get(1, 0),
        ]
    return counts


def plot_dimension_score(anno_t2: pd.DataFrame, out_path: str = "hist/dimensions/dimension_score.pdf") -> None:
    counts = _adequacy_counts_per_system(anno_t2)
    totals = {rt: len(anno_t2[anno_t2["RESPONSE_TYPE"] == rt]) for rt in RESPONSE_TYPES}

    labels = ["Empathy", "Emotion", "Engagement"]
    width = 0.235
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    rect_specs = [
        (-width * 1.1, RESPONSE_TYPES[0]),
        (0,             RESPONSE_TYPES[1]),
        (+width * 1.1, RESPONSE_TYPES[2]),
    ]
    for offset, rt in rect_specs:
        rects = ax.bar(x + offset, counts[rt], width, label=RESPONSE_TYPE_LABEL[rt])
        ax.bar_label(rects, labels=[f"{c/totals[rt]:.1%}" for c in counts[rt]], padding=3)

    ax.set_xlabel("Dimension", fontsize=12)
    ax.set_ylabel("Hit", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="lower center", ncol=3, fancybox=True, fontsize="small",
              handlelength=1.0, handleheight=1.0, columnspacing=1.8, handletextpad=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(x, labels, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.show()
    plt.close(fig)


def _per_level_counts(
    anno_t2: pd.DataFrame, level_column: str, adequacy_column: str
) -> dict[str, list[int]]:
    counts: dict[str, list[int]] = {rt: [] for rt in RESPONSE_TYPES}
    for response_type in RESPONSE_TYPES:
        for level in range(1, 6):
            counts[response_type].append(
                anno_t2[
                    (anno_t2["RESPONSE_TYPE"] == response_type)
                    & (anno_t2[level_column] == level)
                    & (anno_t2[adequacy_column] == 1)
                ].shape[0]
            )
    return counts


def plot_per_level(
    anno_t2: pd.DataFrame,
    level_column: str,
    adequacy_column: str,
    title: str,
) -> None:
    counts = _per_level_counts(anno_t2, level_column, adequacy_column)
    labels = [1, 2, 3, 4, 5]
    width = 0.2
    x = np.arange(len(labels))

    fig, ax = plt.subplots()
    ax.bar(x - width, counts[RESPONSE_TYPES[0]], width, label="SFT")
    ax.bar(x,         counts[RESPONSE_TYPES[1]], width, label="PPO")
    ax.bar(x + width, counts[RESPONSE_TYPES[2]], width, label="DPO")
    ax.set_xlabel(level_column.replace("_", " ").title())
    ax.set_ylabel("Counts")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()


def report_task2(anno_t2: pd.DataFrame) -> None:
    counts = _adequacy_counts_per_system(anno_t2)
    for rt in RESPONSE_TYPES:
        empathy, emotion, engagement = counts[rt]
        print(f"{rt} - EMPATHY_ADEQUACY: {empathy}, EMOTION_ADEQUACY: {emotion}, ENGAGEMENT_ADEQUACY: {engagement}")
    plot_dimension_score(anno_t2)
    plot_per_level(anno_t2, "EMPATHY_LEVEL", "EMPATHY_ADEQUACY",
                   "Counts of 1 in EMPATHY_ADEQUACY for each EMPATHY_LEVEL")
    plot_per_level(anno_t2, "EMOTION_LEVEL", "EMOTION_ADEQUACY",
                   "Counts of 1 in EMOTION_ADEQUACY for each EMOTION_LEVEL")
    plot_per_level(anno_t2, "ENGAGEMENT_LEVEL", "ENGAGEMENT_ADEQUACY",
                   "Counts of 1 in ENGAGEMENT_ADEQUACY for each ENGAGEMENT_LEVEL")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(annotators: list[str] = DEFAULT_ANNOTATORS) -> None:
    data_test_df = pd.read_csv("data/data_test_df.csv")
    anno_t1 = load_annotations(annotators, "task1")
    anno_t2 = load_annotations(annotators, "task2")
    anno_t2 = attach_response_type(anno_t2, data_test_df)
    anno_t2 = normalize_task2(anno_t2)

    print("\n=== TASK 1 ===")
    report_task1(anno_t1)
    print("\n=== TASK 2 ===")
    report_task2(anno_t2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotators", nargs="+", default=DEFAULT_ANNOTATORS)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(annotators=args.annotators)
