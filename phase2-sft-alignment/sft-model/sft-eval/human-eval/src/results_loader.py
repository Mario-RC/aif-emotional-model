"""Load annotator-filled Excel files back into typed DataFrames.

The annotators return spreadsheets with the same row order as the
``data/aux/anno{i}_t{j}_aux.xlsx`` files (which preserved the original
``HUMAN_EMO`` / ``SFT_EMO*`` columns used to compute metrics). We load
both sides and splice the annotated columns into the auxiliary frames
so that downstream metric code can index every piece it needs from a
single DataFrame per ``(annotator, task)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd

from .config import (
    ANNOTATORS, AUX_DIR, MODELS, RESULTS_DIR, TASKS,
    Annotator, Model, Task,
)


@dataclass
class AnnotatorResults:
    """Annotated data for a single annotator across all four tasks."""

    annotator: Annotator
    tasks: Dict[int, pd.DataFrame] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Column plans: which annotator columns to splice into the aux DataFrame
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


def load_annotator(
    annotator: Annotator,
    models: Sequence[Model] = MODELS,
    tasks: Sequence[Task] = TASKS,
    aux_dir: Path = AUX_DIR,
    results_dir: Path = RESULTS_DIR,
) -> AnnotatorResults:
    """Load one annotator's filled-in files and splice them with the aux
    frames."""
    out = AnnotatorResults(annotator=annotator)
    for task in tasks:
        aux_path = aux_dir / f"anno{annotator.index}_t{task.num}_aux.xlsx"
        filled_path = (results_dir / annotator.name
                       / f"anno{annotator.index}_{task.slug}.xlsx")
        aux_df = pd.read_excel(aux_path)
        filled_df = pd.read_excel(filled_path)
        for position, col in _splice_plan(task, models):
            aux_df.insert(position, col, filled_df[col])
        out.tasks[task.num] = aux_df
    return out


def load_all_annotators(
    annotators: Sequence[Annotator] = ANNOTATORS,
    **kwargs,
) -> list[AnnotatorResults]:
    """Load every *included* annotator. Excluded annotators (see
    ``Annotator.included``) are silently skipped so downstream metric
    code only operates on the participating subset."""
    return [load_annotator(a, **kwargs) for a in annotators if a.included]
