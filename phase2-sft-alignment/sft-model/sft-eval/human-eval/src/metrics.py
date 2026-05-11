"""Metrics computed over annotator-filled DataFrames.

Groups the three metric families found in
``human_annotations_results.ipynb``:

* **Rank@K** (Tasks 1 and 3) — fraction of rows where a model received
  a quality rank ≤ ``K``.
* **Mean / stdev** (Tasks 1, 3, 4) — basic descriptive statistics over
  the numeric quality ratings.
* **Emotion hit counts** (Task 2) — how often the annotator's emotion
  label matches the ground-truth emotion for both the user prompt and
  each model's second response.
* **Inter-annotator agreement (Krippendorff alpha)** over the shared
  IAA rows for each task.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Dict, Iterable, List, Sequence

import pandas as pd
from krippendorff import alpha

from .config import MODELS, Model
from .results_loader import AnnotatorResults


# ---------------------------------------------------------------------------
# Rank@K over Tasks 1 and 3
# ---------------------------------------------------------------------------

def rank_at_k(series: pd.Series, k: int) -> float:
    """Fraction of rows whose rank is 1..K."""
    values = series.dropna().astype(int)
    if len(values) == 0:
        return 0.0
    return float((values <= k).sum()) / len(values)


@dataclass
class RankAtKTable:
    """Rank@K for every model across two tasks, for one annotator."""

    annotator_name: str
    k: int
    task_a: int     # task number of first column, e.g. 1
    task_b: int     # task number of second column, e.g. 3
    rows: Dict[str, tuple[float, float]] = field(default_factory=dict)

    def add(self, model_label: str, a: float, b: float) -> None:
        self.rows[model_label] = (a, b)

    def render(self) -> str:
        head = (f"{self.annotator_name} @{self.k} - "
                f"Task {self.task_a} - Task {self.task_b}")
        body = "\n".join(
            f"{label:<8} - {a:.3f}  - {b:.3f}"
            for label, (a, b) in self.rows.items()
        )
        return f"{head}\n{body}"


def compute_rank_at_k(
    result: AnnotatorResults,
    k: int,
    task_a: int = 1, task_b: int = 3,
    col_a: str = "EMPATHY_QUALITY",
    col_b: str = "QUESTION_QUALITY",
    models: Sequence[Model] = MODELS,
) -> RankAtKTable:
    table = RankAtKTable(
        annotator_name=f"ANNO {result.annotator.index}",
        k=k, task_a=task_a, task_b=task_b,
    )
    df_a = result.tasks[task_a]
    df_b = result.tasks[task_b]
    for i, model in enumerate(models, start=1):
        a = rank_at_k(df_a[f"MODEL{i}_{col_a}"], k)
        b = rank_at_k(df_b[f"MODEL{i}_{col_b}"], k)
        table.add(model.label, a, b)
    return table


# ---------------------------------------------------------------------------
# Mean / stdev across Tasks 1, 3, 4
# ---------------------------------------------------------------------------

@dataclass
class MeanStdTable:
    annotator_name: str
    # Per model: {'Task 1': (mean, std), 'Task 3': (mean, std), 'Task 4': (mean, std)}
    rows: Dict[str, Dict[str, tuple[float, float]]] = field(default_factory=dict)

    def add(self, model_label: str, task_label: str, mean_: float, std: float) -> None:
        self.rows.setdefault(model_label, {})[task_label] = (mean_, std)

    def render(self) -> str:
        lines = [f"{self.annotator_name}       -- Task 1 -- Task 3 -- Task 4"]
        for label, stats in self.rows.items():
            t1_m, t1_s = stats.get("Task 1", (float("nan"), float("nan")))
            t3_m, t3_s = stats.get("Task 3", (float("nan"), float("nan")))
            t4_m, t4_s = stats.get("Task 4", (float("nan"), float("nan")))
            lines.append(f"\n{label} MEAN   -- {t1_m:.3f}  -- {t3_m:.3f}  -- {t4_m:.3f}")
            lines.append(f"{label} STD    -- {t1_s:.3f}  -- {t3_s:.3f}  -- {t4_s:.3f}")
        return "\n".join(lines)


_MEAN_STD_SPEC = (
    ("Task 1", 1, "EMPATHY_QUALITY"),
    ("Task 3", 3, "QUESTION_QUALITY"),
    ("Task 4", 4, "OVERALL_QUALITY"),
)


def compute_mean_std(
    result: AnnotatorResults, models: Sequence[Model] = MODELS,
) -> MeanStdTable:
    table = MeanStdTable(annotator_name=f"ANNO {result.annotator.index}")
    for label, task_num, col in _MEAN_STD_SPEC:
        df = result.tasks[task_num]
        for i, model in enumerate(models, start=1):
            values = df[f"MODEL{i}_{col}"].dropna().astype(float).tolist()
            if len(values) < 2:
                table.add(model.label, label, float("nan"), float("nan"))
                continue
            table.add(model.label, label, mean(values), stdev(values))
    return table


# ---------------------------------------------------------------------------
# Overall Rank@K / Task-4 summary table
# ---------------------------------------------------------------------------

PUBLISHED_OVERALL_QUALITY: Dict[
    str, tuple[float, float, float, float, float, float]
] = {
    "GLM4": (0.27, 0.48, 0.29, 0.37, 3.43, 0.77),
    "Gemma2": (0.14, 0.36, 0.24, 0.40, 3.48, 0.71),
    "LLaMA3": (0.21, 0.38, 0.13, 0.43, 3.64, 0.35),
    "Mistral": (0.16, 0.43, 0.15, 0.39, 3.68, 0.32),
    "Phi3": (0.21, 0.36, 0.19, 0.41, 3.57, 0.67),
}


@dataclass
class OverallQualityTable:
    """Overall table used in the manuscript-style human-eval summary."""

    rows: Dict[str, tuple[float, float, float, float, float, float]] = field(
        default_factory=dict
    )

    def add(
        self, model_label: str, task1_p1: float, task1_p2: float,
        task3_p1: float, task3_p2: float, task4_mean: float,
        task4_std: float,
    ) -> None:
        self.rows[model_label] = (
            task1_p1, task1_p2, task3_p1, task3_p2, task4_mean, task4_std,
        )

    def render(self) -> str:
        lines = [
            "OVERALL       -- Task 1 P@1 -- Task 1 P@2 -- "
            "Task 3 P@1 -- Task 3 P@2 -- Task 4 Mean ± Std"
        ]
        for label, row in self.rows.items():
            t1_p1, t1_p2, t3_p1, t3_p2, t4_m, t4_s = row
            lines.append(
                f"{label:<8} - {t1_p1:.2f} - {t1_p2:.2f} - "
                f"{t3_p1:.2f} - {t3_p2:.2f} - {t4_m:.2f} ± {t4_s:.2f}"
            )
        return "\n".join(lines)


def compute_overall_quality(
    results: Sequence[AnnotatorResults],
    models: Sequence[Model] = MODELS,
    published_rows: (
        Dict[str, tuple[float, float, float, float, float, float]] | None
    ) = None,
) -> OverallQualityTable:
    """Aggregate Rank@K and Task-4 quality across all annotators.

    Passing ``published_rows`` renders a fixed aggregate table instead of
    recomputing it directly from the anonymized annotator spreadsheets.
    """
    if not results:
        raise ValueError("Need at least one annotator's results to aggregate")

    table = OverallQualityTable()
    if published_rows is not None:
        for model in models:
            table.add(model.label, *published_rows[model.label])
        return table

    for i, model in enumerate(models, start=1):
        t1_p1 = mean([
            rank_at_k(res.tasks[1][f"MODEL{i}_EMPATHY_QUALITY"], 1)
            for res in results
        ])
        t1_p2 = mean([
            rank_at_k(res.tasks[1][f"MODEL{i}_EMPATHY_QUALITY"], 2)
            for res in results
        ])
        t3_p1 = mean([
            rank_at_k(res.tasks[3][f"MODEL{i}_QUESTION_QUALITY"], 1)
            for res in results
        ])
        t3_p2 = mean([
            rank_at_k(res.tasks[3][f"MODEL{i}_QUESTION_QUALITY"], 2)
            for res in results
        ])
        task4_means = [
            mean(
                res.tasks[4][f"MODEL{i}_OVERALL_QUALITY"]
                .dropna().astype(float).tolist()
            )
            for res in results
        ]
        task4_mean = mean(task4_means)
        task4_std = stdev(task4_means) if len(task4_means) > 1 else 0.0
        table.add(model.label, t1_p1, t1_p2, t3_p1, t3_p2,
                  task4_mean, task4_std)
    return table


# ---------------------------------------------------------------------------
# Task 2: emotion hit counts
# ---------------------------------------------------------------------------

@dataclass
class EmotionHits:
    """Task-2 emotion-labeling accuracy for one annotator."""

    annotator_name: str
    user_hits: int = 0
    model_hits: List[int] = field(default_factory=list)
    hit_emotions_by_source: Dict[str, List[str]] = field(default_factory=dict)

    def total_hit_counts(self) -> List[int]:
        """``[user, model1, model2, ...]`` for plotting."""
        return [self.user_hits] + list(self.model_hits)

    def all_emotions_flat(self) -> Counter:
        return Counter(
            e for emotions in self.hit_emotions_by_source.values() for e in emotions
        )


def _count_matches(df: pd.DataFrame, truth_col: str, pred_col: str) -> tuple[int, List[str]]:
    hits = [p for t, p in zip(df[truth_col], df[pred_col])
            if isinstance(t, str) and isinstance(p, str) and t.lower() == p.lower()]
    return len(hits), hits


def compute_emotion_hits(
    result: AnnotatorResults, models: Sequence[Model] = MODELS,
) -> EmotionHits:
    if 2 not in result.tasks:
        raise ValueError(f"ANNO {result.annotator.index} has no Task 2 annotations")

    df = result.tasks[2]
    user_hits, user_tags = _count_matches(df, "HUMAN_EMO", "USER_EMOTION")
    out = EmotionHits(annotator_name=f"ANNO {result.annotator.index}",
                      user_hits=user_hits)
    out.hit_emotions_by_source["HUMAN"] = user_tags
    for i, model in enumerate(models, start=1):
        n, tags = _count_matches(
            df, f"SFT_EMO2_{model.short}", f"MODEL{i}_RESPONSE_EMOTION",
        )
        out.model_hits.append(n)
        out.hit_emotions_by_source[model.label] = tags
    return out


# ---------------------------------------------------------------------------
# Cross-annotator aggregation for Task 2
# ---------------------------------------------------------------------------

@dataclass
class OverallEmotionHits:
    sums: List[int]           # total across annotators per column
    means: List[float]
    stds: List[float]
    per_emotion: Counter
    per_emotion_mean: Dict[str, float]
    per_emotion_std: Dict[str, float]


def aggregate_emotion_hits(annos: Sequence[EmotionHits]) -> OverallEmotionHits:
    if not annos:
        raise ValueError("Need at least one annotator's results to aggregate")

    cols = len(annos[0].total_hit_counts())
    per_col = [[a.total_hit_counts()[i] for a in annos] for i in range(cols)]
    sums = [sum(col) for col in per_col]
    means = [mean(col) for col in per_col]
    stds = [stdev(col) if len(col) > 1 else 0.0 for col in per_col]

    # Per-emotion aggregation using the same semantics as the original
    # notebook: sum each annotator counter, divide by the number of
    # annotators, and treat missing emotions as 0 for standard deviation.
    per_emotion = Counter()
    per_annotator_emotions: List[Counter] = []
    for a in annos:
        c = a.all_emotions_flat()
        per_annotator_emotions.append(c)
        per_emotion.update(c)

    mean_ = {e: (n / len(annos)) for e, n in per_emotion.items()}
    std_ = {
        e: stdev([c.get(e, 0) for c in per_annotator_emotions])
        if len(per_annotator_emotions) > 1 else 0.0
        for e in mean_.keys()
    }
    return OverallEmotionHits(sums, means, stds, per_emotion, mean_, std_)


# ---------------------------------------------------------------------------
# Inter-annotator agreement (Krippendorff alpha) over the IAA rows
# ---------------------------------------------------------------------------

_IAA_SPEC = {
    1: ("EMPATHY_QUALITY",   "ordinal"),
    2: ("RESPONSE_EMOTION",  "nominal"),   # Task 2: emotion label
    3: ("QUESTION_QUALITY",  "ordinal"),
    4: ("OVERALL_QUALITY",   "ordinal"),
}


def _iaa_column_for(task_num: int, i: int) -> str:
    base = _IAA_SPEC[task_num][0]
    return f"MODEL{i}_{base}"


def _iaa_flat_values(df: pd.DataFrame, task_num: int,
                     models: Sequence[Model]) -> List:
    columns = [_iaa_column_for(task_num, i) for i in range(1, len(models) + 1)]
    values: List = []
    for col in columns:
        values.extend(df[col].to_list())
    return values


def compute_iaa_alpha(
    annos: Sequence[AnnotatorResults],
    iaa_uids_by_task: Dict[int, Iterable[str]],
    models: Sequence[Model] = MODELS,
) -> Dict[int, float]:
    """Krippendorff alpha for each of the 4 tasks, restricted to the
    shared IAA rows."""
    out: Dict[int, float] = {}
    for task_num, (_, level) in _IAA_SPEC.items():
        task_annos = [a for a in annos if task_num in a.tasks]
        if len(task_annos) < 2:
            out[task_num] = float("nan")
            continue

        per_annotator = []
        uids = list(iaa_uids_by_task[task_num])
        common_uids = set(uids)
        for a in task_annos:
            common_uids &= set(a.tasks[task_num]["UID"])
        ordered_uids = [uid for uid in uids if uid in common_uids]
        if not ordered_uids:
            raise ValueError(f"No shared IAA UIDs found for task {task_num}")

        for a in task_annos:
            df = a.tasks[task_num].set_index("UID")
            iaa_df = df.loc[ordered_uids].reset_index()
            per_annotator.append(_iaa_flat_values(iaa_df, task_num, models))
        try:
            out[task_num] = float(alpha(
                per_annotator, level_of_measurement=level,
            ))
        except ValueError as exc:
            if "more than one value in the domain" not in str(exc):
                raise
            out[task_num] = float("nan")
    return out


def iaa_uids_for(task_num: int, base_uids: Sequence[str]) -> List[str]:
    """Append the task suffix to every base UID (``SFTANNO-000006`` →
    ``SFTANNO-000006-0002`` for task 2). If a UID already has the task
    suffix it is returned unchanged."""
    suffix = f"-{task_num:04d}"
    return [u if u.endswith(suffix) else u + suffix for u in base_uids]
