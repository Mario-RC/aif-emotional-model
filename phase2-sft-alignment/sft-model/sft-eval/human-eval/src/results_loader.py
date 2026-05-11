"""Load annotator-filled Excel files back into typed DataFrames.

For ranking/quality tasks the annotator files already contain every
column needed by the metrics. Task 2 also needs the hidden emotion-label
answers, so we recover them from ``data/task2_reference_labels.xlsx`` by
matching the visible prompt/response text. Generated ``data/aux`` or
``data/tasks`` files are not a safe fallback for old submitted
annotations because they can be refreshed from a different prediction run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd

from .config import (
    ANNOTATORS, MODELS, RESULTS_DIR, TASK2_REFERENCE_LABELS_PATH, TASKS,
    Annotator, Model, Task,
)


@dataclass
class AnnotatorResults:
    """Annotated data for a single annotator across all four tasks."""

    annotator: Annotator
    tasks: Dict[int, pd.DataFrame] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Splice annotator-visible columns into internal task layouts
# ---------------------------------------------------------------------------

def _splice_plan(task: Task, models: Sequence[Model]) -> list[tuple[int, str]]:
    """Column insertion plan mirroring the original notebook's layout.

    Returns ``[(insert_position, column_name), ...]`` in insertion order.
    Positions assume the aux DataFrame has the original columns (UID,
    SID, HUMAN_UTT, HUMAN_EMO, SFT_R*_{SHORT}, SFT_EMO*_{SHORT}, TARGET_*,
    ...) so insertions happen in the right slots.
    """
    plan: list[tuple[int, str]] = [(3, "USER_PROMPT")]
    if task.num == 2:
        plan.append((5, "USER_EMOTION"))
        # Response and emotion for each model, 4 cells apart
        for i, _ in enumerate(models, start=1):
            base = 7 + 4 * (i - 1)
            plan.append((base, f"MODEL{i}_RESPONSE"))
            plan.append((base + 2, f"MODEL{i}_RESPONSE_EMOTION"))
    elif task.num == 4:
        # Task 4 has response + overall-quality for each model, 6 cells apart.
        for i, _ in enumerate(models, start=1):
            base = 6 + 6 * (i - 1)
            plan.append((base, f"MODEL{i}_RESPONSE"))
            plan.append((base + 1, f"MODEL{i}_OVERALL_QUALITY"))
    else:
        # Tasks 1 and 3 have response + quality for each model, 4 cells apart.
        for i, _ in enumerate(models, start=1):
            base = 6 + 4 * (i - 1)
            plan.append((base, f"MODEL{i}_RESPONSE"))
            plan.append((base + 1, f"MODEL{i}_{task.quality_col}"))
    return plan


def _norm_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _norm_emotion(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    return str(value).strip().title()


def _add_task2_truth_from_table(
    filled_df: pd.DataFrame,
    truth_path: Path = TASK2_REFERENCE_LABELS_PATH,
    models: Sequence[Model] = MODELS,
) -> pd.DataFrame:
    """Attach Task-2 gold emotion labels using the active long truth table.

    The truth table stores one row per prompt/model response. The
    annotator spreadsheets store one row per prompt with five model
    responses, so matching on visible text is the durable join key.
    """
    truth_df = pd.read_excel(truth_path)
    required = {
        "PROMPT", "PROMPT_EMOTION", "SFT_RESPONSE", "SFT_EMOTION",
    }
    missing = sorted(required - set(truth_df.columns))
    if missing:
        raise ValueError(f"{truth_path} is missing columns: {missing}")

    truth_df = truth_df.copy()
    truth_df["_PROMPT_KEY"] = truth_df["PROMPT"].map(_norm_text)
    truth_df["_RESPONSE_KEY"] = truth_df["SFT_RESPONSE"].map(_norm_text)
    prompt_truth = (
        truth_df.drop_duplicates("_PROMPT_KEY")
        .set_index("_PROMPT_KEY")["PROMPT_EMOTION"]
        .to_dict()
    )
    response_truth = (
        truth_df.drop_duplicates(["_PROMPT_KEY", "_RESPONSE_KEY"])
        .set_index(["_PROMPT_KEY", "_RESPONSE_KEY"])["SFT_EMOTION"]
        .to_dict()
    )

    out = filled_df.copy()
    out["HUMAN_EMO"] = [
        _norm_emotion(prompt_truth.get(_norm_text(prompt)))
        for prompt in out["USER_PROMPT"]
    ]
    missing_prompt_count = int(out["HUMAN_EMO"].isna().sum())

    missing_response_counts: dict[str, int] = {}
    for i, model in enumerate(models, start=1):
        response_col = f"MODEL{i}_RESPONSE"
        truth_col = f"SFT_EMO2_{model.short}"
        values = []
        for prompt, response in zip(out["USER_PROMPT"], out[response_col]):
            key = (_norm_text(prompt), _norm_text(response))
            if key not in response_truth:
                missing_response_counts[response_col] = (
                    missing_response_counts.get(response_col, 0) + 1
                )
            values.append(_norm_emotion(response_truth.get(key)))
        out[truth_col] = values

    if missing_prompt_count or missing_response_counts:
        response_summary = ", ".join(
            f"{col}: {len(out) - n}/{len(out)} matched"
            for col, n in sorted(missing_response_counts.items())
        ) or "all response columns matched"
        raise ValueError(
            f"Could not recover all Task-2 reference labels from {truth_path}. "
            f"Prompt matches: {len(out) - missing_prompt_count}/{len(out)}. "
            f"Response pair matches: {response_summary}. "
            "This usually means the Task-2 reference-label file was generated "
            "from a different prediction run than the filled annotator results."
        )
    return out


def load_annotator(
    annotator: Annotator,
    models: Sequence[Model] = MODELS,
    tasks: Sequence[Task] = TASKS,
    results_dir: Path = RESULTS_DIR,
) -> AnnotatorResults:
    """Load one annotator's filled-in files."""
    out = AnnotatorResults(annotator=annotator)
    for task in tasks:
        if task.num not in annotator.task_nums:
            continue

        filled_path = (results_dir / annotator.name
                       / f"anno{annotator.index}_{task.slug}.xlsx")
        filled_df = pd.read_excel(filled_path)

        if task.num == 2:
            if not TASK2_REFERENCE_LABELS_PATH.exists():
                raise FileNotFoundError(
                    f"{TASK2_REFERENCE_LABELS_PATH} is required to analyze Task 2. "
                    "Refusing to use regenerated data/tasks files because "
                    "they can silently change the published human-eval "
                    "metrics. Backup folders are intentionally ignored by "
                    "the pipeline."
                )
            out.tasks[task.num] = _add_task2_truth_from_table(
                filled_df, TASK2_REFERENCE_LABELS_PATH, models,
            )
        else:
            out.tasks[task.num] = filled_df
    return out


def load_all_annotators(
    annotators: Sequence[Annotator] = ANNOTATORS,
    **kwargs,
) -> list[AnnotatorResults]:
    """Load every included annotator and only the tasks assigned to them."""
    return [load_annotator(a, **kwargs) for a in annotators if a.included]
