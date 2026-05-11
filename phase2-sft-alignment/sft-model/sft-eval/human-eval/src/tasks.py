"""Build the four task DataFrames, split them across annotators and
rewrite them into the schema seen by annotators.

Replaces the ``Create Tasks`` / ``Split Turns`` / ``Give Format``
sections of ``human_annotations_generation.ipynb`` with a handful of
small functions that parametrize over task and annotator rather than
copy-pasting per-annotator blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from .config import (
    ANNOTATORS, AUX_DIR, FINAL_DIR, MODELS, SAMPLING,
    TASK2_REFERENCE_LABELS_PATH, TASKS, TASKS_DIR, Annotator, Model, Task,
    task_uid_suffix,
)


# ---------------------------------------------------------------------------
# 1. Build per-task DataFrames from the wide SFT DataFrame
# ---------------------------------------------------------------------------

def build_task_dataframe(wide: pd.DataFrame, task: Task,
                         models: Sequence[Model] = MODELS) -> pd.DataFrame:
    """Create the DataFrame for a single task.

    For tasks 1–3 the response columns are taken directly from
    ``SFT_R{n}_<SHORT>``; for task 4 the three response sentences are
    concatenated into a single ``SFT_R_<SHORT>`` column.
    """
    suffix = task.response_col_suffix  # "R1" / "R2" / "R3" / ""
    df = pd.DataFrame()
    df["UID"] = wide["UID"] + task_uid_suffix(task.num)
    df["SID"] = task.sid
    df["HUMAN_UTT"] = wide["HUMAN_UTT"]
    df["HUMAN_EMO"] = wide["HUMAN_EMO"]

    if task.num == 4:
        # Response = R1 + R2 + R3 joined with spaces; all three emotions kept.
        for model in models:
            parts = [wide[f"SFT_R{i}_{model.short}"] for i in (1, 2, 3)]
            df[f"SFT_R_{model.short}"] = parts[0] + " " + parts[1] + " " + parts[2]
            for i in (1, 2, 3):
                df[f"SFT_EMO{i}_{model.short}"] = wide[f"SFT_EMO{i}_{model.short}"]
        df["TARGET_R"] = wide["TARGET_R1"] + " " + wide["TARGET_R2"] + " " + wide["TARGET_R3"]
        for i in (1, 2, 3):
            df[f"TARGET_EMO{i}"] = wide[f"TARGET_EMO{i}"]
        return df

    for model in models:
        df[f"SFT_{suffix}_{model.short}"] = wide[f"SFT_{suffix}_{model.short}"]
        df[f"SFT_EMO{suffix[-1]}_{model.short}"] = wide[f"SFT_EMO{suffix[-1]}_{model.short}"]
    df[f"TARGET_{suffix}"] = wide[f"TARGET_{suffix}"]
    df[f"TARGET_EMO{suffix[-1]}"] = wide[f"TARGET_EMO{suffix[-1]}"]
    return df


def save_task_dataframes(dfs: Dict[int, pd.DataFrame],
                         tasks_dir: Path = TASKS_DIR) -> None:
    tasks_dir.mkdir(parents=True, exist_ok=True)
    for num, df in dfs.items():
        df.to_excel(tasks_dir / f"task{num}.xlsx", index=False)


def build_task2_reference_labels(
    wide: pd.DataFrame, models: Sequence[Model] = MODELS,
) -> pd.DataFrame:
    """Build the long Task-2 reference-label table.

    The annotator-facing Task 2 file is wide: one row per prompt, with
    five model responses. The analysis also needs a long reference table
    with one row per prompt/model response. That table is
    the canonical source for the hidden prompt/response emotion labels
    when analyzing filled-in annotations.
    """
    columns = [
        "UID", "SID", "MODEL", "PROMPT", "PROMPT_EMOTION",
        "SFT_RESPONSE", "SFT_EMOTION", "TARGET_RESPONSE", "TARGET_EMOTION",
    ]
    rows = []
    wide = wide.reset_index(drop=True)
    n_rows = len(wide)

    for model_idx, model in enumerate(models):
        for row_idx, row in wide.iterrows():
            rows.append({
                "UID": f"SFTANNO-{model_idx * n_rows + row_idx:06d}-0002",
                "SID": "P-R2",
                "MODEL": model.name,
                "PROMPT": row["HUMAN_UTT"],
                "PROMPT_EMOTION": row["HUMAN_EMO"],
                "SFT_RESPONSE": row[f"SFT_R2_{model.short}"],
                "SFT_EMOTION": row[f"SFT_EMO2_{model.short}"],
                "TARGET_RESPONSE": row["TARGET_R2"],
                "TARGET_EMOTION": row["TARGET_EMO2"],
            })
    return pd.DataFrame(rows, columns=columns)


def save_task2_reference_labels(
    df: pd.DataFrame, path: Path = TASK2_REFERENCE_LABELS_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False)


# ---------------------------------------------------------------------------
# 2. Per-annotator sampling
# ---------------------------------------------------------------------------

@dataclass
class AnnotatorSlice:
    """Per-annotator subsets of the tasks assigned to this annotator."""

    annotator: Annotator
    tasks: Dict[int, pd.DataFrame]


def _sample_per_emotion(
    df: pd.DataFrame, n: int, random_state: int,
) -> tuple[pd.DataFrame, pd.Index]:
    """Sample ``n`` rows of every emotion from ``df`` (``HUMAN_EMO``
    column). Returns the picked rows (index-reset) and the original
    row indices so the caller can drop them from the source pool.
    """
    picked = df.groupby("HUMAN_EMO").apply(
        lambda x: x.sample(n=n, random_state=random_state)
    )
    return picked.reset_index(drop=True), picked.index.get_level_values(1)


def split_across_annotators(
    task_dfs: Dict[int, pd.DataFrame],
    annotators: Sequence[Annotator] = ANNOTATORS,
) -> List[AnnotatorSlice]:
    """Replicate the human-evaluation sampling protocol.

    1. Grab the first ``SAMPLING.iaa_size`` rows whose user emotion
       matches the chatbot's R2 emotion (the IAA set — shared).
    2. For every annotator, draw
       ``SAMPLING.per_emotion_shared`` rows per emotion from what's left.
       These rows are DROPPED from the pool so each annotator gets a
       disjoint slice.
    3. Task 2 gets an extra
       ``SAMPLING.per_emotion_task2_extra`` rows per emotion and per
       annotator, plus the leftover task-2 rows distributed 3-per-
       annotator.
    4. Each per-annotator DataFrame is shuffled with a fixed seed.
    """
    active_annotators = [annotator for annotator in annotators if annotator.included]
    pools = {num: df.copy() for num, df in task_dfs.items()}

    # --- IAA (shared rows) -------------------------------------------------
    t2_pool = pools[2]
    iaa_idx = t2_pool[t2_pool["HUMAN_EMO"] == t2_pool["TARGET_EMO2"]].head(
        SAMPLING.iaa_size
    ).index

    iaa_slices = {num: df.iloc[iaa_idx].reset_index(drop=True)
                  for num, df in pools.items()}
    for num, df in pools.items():
        pools[num] = df.drop(df.index[iaa_idx]).reset_index(drop=True)

    # --- 2-per-emotion (disjoint across annotators, aligned across tasks) --
    slices: List[AnnotatorSlice] = []
    for annotator in active_annotators:
        # Sample on task 1 only; the picked indices are used for every task
        # so the same dialogues appear in this annotator's t1/t2/t3/t4.
        _, picked_idx = _sample_per_emotion(
            pools[1], SAMPLING.per_emotion_shared, SAMPLING.sampling_random_state,
        )
        picked_idx_list = list(picked_idx)
        per_task: Dict[int, pd.DataFrame] = {}
        for num, df in pools.items():
            if num not in annotator.task_nums:
                continue

            extra = df.iloc[picked_idx_list].reset_index(drop=True)
            per_task[num] = pd.concat(
                [iaa_slices[num], extra], axis=0, ignore_index=True,
            )
            pools[num] = df.drop(df.index[picked_idx_list]).reset_index(drop=True)
        slices.append(AnnotatorSlice(annotator=annotator, tasks=per_task))

    # --- Task-2 extra rows -------------------------------------------------
    task2_slices = [slice_ for slice_ in slices if 2 in slice_.tasks]
    for slice_ in task2_slices:
        extra, picked_idx = _sample_per_emotion(
            pools[2], SAMPLING.per_emotion_task2_extra, SAMPLING.sampling_random_state,
        )
        slice_.tasks[2] = pd.concat(
            [slice_.tasks[2], extra], axis=0, ignore_index=True,
        )
        pools[2] = pools[2].drop(pools[2].index[list(picked_idx)]).reset_index(drop=True)

    # --- Task-2 leftover: first n-1 annotators get ceil(L/n), last takes
    #     whatever remains (matches the original notebook's distribution).
    leftover = pools[2]
    n = len(task2_slices)
    if n:
        chunk = -(-len(leftover) // n)  # ceil division
        for i, slice_ in enumerate(task2_slices):
            start = min(i * chunk, len(leftover))
            stop = len(leftover) if i == n - 1 else min(start + chunk, len(leftover))
            slice_.tasks[2] = pd.concat(
                [slice_.tasks[2], leftover.iloc[start:stop]],
                axis=0, ignore_index=True,
            )

    # --- Shuffle -----------------------------------------------------------
    for slice_ in slices:
        for num, df in slice_.tasks.items():
            slice_.tasks[num] = df.sample(
                frac=1, random_state=SAMPLING.shuffle_random_state,
            ).reset_index(drop=True)

    return slices


def save_aux_slices(slices: Sequence[AnnotatorSlice],
                    aux_dir: Path = AUX_DIR) -> None:
    """Persist the per-annotator dataframes with internal columns intact."""
    aux_dir.mkdir(parents=True, exist_ok=True)
    for slice_ in slices:
        i = slice_.annotator.index
        for num, df in slice_.tasks.items():
            df.to_excel(aux_dir / f"anno{i}_t{num}_aux.xlsx", index=False)


# ---------------------------------------------------------------------------
# 3. Reformat each slice into the schema seen by annotators
# ---------------------------------------------------------------------------

def format_for_annotator(
    aux_df: pd.DataFrame, task: Task, models: Sequence[Model] = MODELS,
) -> pd.DataFrame:
    """Drop internal columns, rename model columns to ``MODEL{i}_*`` and
    add the empty annotation columns to fill in.
    """
    df = aux_df.copy()
    # Normalize the prompt column name.
    df = df.rename(columns={"HUMAN_UTT": "USER_PROMPT"})

    # Drop internals shared across all tasks.
    drops = ["SID", "HUMAN_EMO"]
    if task.num == 4:
        drops += [f"SFT_EMO{i}_{m.short}" for m in models for i in (1, 2, 3)]
        drops += ["TARGET_R", "TARGET_EMO1", "TARGET_EMO2", "TARGET_EMO3"]
    else:
        i = task.response_col_suffix[-1]
        drops += [f"SFT_EMO{i}_{m.short}" for m in models]
        drops += [f"TARGET_R{i}", f"TARGET_EMO{i}"]
    df = df.drop(columns=drops)

    # Rename SFT_R*_{SHORT} → MODEL{i}_RESPONSE.
    for i, m in enumerate(models, start=1):
        if task.num == 4:
            df = df.rename(columns={f"SFT_R_{m.short}": f"MODEL{i}_RESPONSE"})
        else:
            suffix = task.response_col_suffix
            df = df.rename(columns={f"SFT_{suffix}_{m.short}": f"MODEL{i}_RESPONSE"})

    # Add empty annotation columns.
    if task.num == 2:
        df.insert(2, "USER_EMOTION", "")
        # Emotion column follows each MODEL{i}_RESPONSE column (positions 4, 6, ...).
        for i in range(1, len(models) + 1):
            insert_at = 2 + 2 * i  # 4, 6, 8, 10, 12
            df.insert(insert_at, f"MODEL{i}_RESPONSE_EMOTION", "")
        df["COMMENTS"] = ""
    else:
        for i in range(1, len(models) + 1):
            insert_at = 2 * i + 1  # 3, 5, 7, 9, 11
            df.insert(insert_at, f"MODEL{i}_{task.quality_col}", "")
        df["COMMENTS"] = ""

    return df


def save_final_slices(slices: Sequence[AnnotatorSlice],
                      final_dir: Path = FINAL_DIR,
                      models: Sequence[Model] = MODELS) -> None:
    """Write the annotator-facing Excel files to ``data/final/``."""
    final_dir.mkdir(parents=True, exist_ok=True)
    task_by_num = {t.num: t for t in TASKS}
    for slice_ in slices:
        i = slice_.annotator.index
        for num, df in slice_.tasks.items():
            task = task_by_num[num]
            formatted = format_for_annotator(df, task, models=models)
            formatted.to_excel(
                final_dir / f"anno{i}_{task.slug}.xlsx", index=False,
            )
