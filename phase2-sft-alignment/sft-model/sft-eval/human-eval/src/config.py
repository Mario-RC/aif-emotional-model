"""Static configuration shared across the human-eval pipeline.

Everything that was a magic constant scattered across the two notebooks —
model names, annotator roster, task schema, file locations, column
layouts — lives here in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
HUMAN_EVAL_DIR = PACKAGE_DIR.parent

DATA_DIR = HUMAN_EVAL_DIR / "data"
TASKS_DIR = DATA_DIR / "tasks"
AUX_DIR = DATA_DIR / "aux"
FINAL_DIR = DATA_DIR / "final"

RESULTS_DIR = HUMAN_EVAL_DIR / "results"
HIST_DIR = HUMAN_EVAL_DIR / "hist"

# SFT prediction JSONs live inside the Phase 2 LLaMA-Factory workspace.
SFT_SAVES_DIR = HUMAN_EVAL_DIR.parent / "sft-model" / "sft-llama-factory-legacy" / "saves"
SFT_RESULTS_FILENAME = "emotional_balanced/demonstration_data_emotional_balanced_test_results.json"

WIDE_CSV_PATH = DATA_DIR / "demonstration_data_emotional_balanced_test_results_dial.csv"


# ---------------------------------------------------------------------------
# Models under evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Model:
    """A model evaluated in the human-annotation study.

    Attributes:
        name: HuggingFace-style identifier, used as a folder name under
            ``saves/`` and to build the SFT results path.
        short: Uppercase alias used as a suffix throughout the wide
            DataFrame column names (e.g. ``SFT_R1_GLM4``).
        label: Pretty label used in plots and printed reports.
    """

    name: str
    short: str
    label: str


MODELS: Tuple[Model, ...] = (
    Model("glm-4-9b-chat-1m",           "GLM4",    "GLM4"),
    Model("gemma-2-9b-it",              "GEMMA",   "Gemma2"),
    Model("Meta-Llama-3-8B-Instruct",   "LLAMA3",  "LLaMA3"),
    Model("Mistral-7B-Instruct-v0.3",   "MISTRAL", "Mistral"),
    Model("Phi-3-small-8k-instruct",    "PHI3",    "Phi3"),
)

MODEL_LABELS: List[str] = ["Human"] + [m.label for m in MODELS]
PRINT_LABELS: List[str] = ["HUMAN"] + [m.label.upper() for m in MODELS]


# ---------------------------------------------------------------------------
# Emotions
# ---------------------------------------------------------------------------

EMOTIONS: Tuple[str, ...] = (
    "Neutral", "Happiness", "Sadness", "Anger", "Surprise", "Fear", "Disgust",
)

EMOTION_COLORS: Dict[str, str] = {
    "Neutral":   "#d3d3d3",
    "Happiness": "#ffff99",
    "Sadness":   "#add8e6",
    "Anger":     "#ff9999",
    "Surprise":  "#ffd700",
    "Fear":      "#dda0dd",
    "Disgust":   "#98fb98",
}


# ---------------------------------------------------------------------------
# Annotators
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Annotator:
    index: int          # 1-based slot (anno1 … anno5)
    name: str           # folder name under results/
    included: bool      # whether the annotator's results are used in aggregation


ANNOTATORS: Tuple[Annotator, ...] = (
    Annotator(1, "kallirroi",    included=False),
    Annotator(2, "marcos",       included=True),
    Annotator(3, "jaime",        included=True),
    Annotator(4, "luisfernando", included=True),
    Annotator(5, "mario",        included=True),
)


# ---------------------------------------------------------------------------
# Task schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """One of the four human-eval tasks.

    Attributes:
        num: Task number (1..4), matches the ``tX`` suffix in DataFrame names.
        slug: Lowercase identifier used in final Excel filenames.
        sid: Segment id stamped on each row (``P-R1``, ``P-R``, …).
        response_col_suffix: Suffix of the per-model response column in the
            wide DataFrame (``R1``, ``R2``, ``R3`` or empty for the full
            concatenation used in task 4).
        quality_col: Name of the empty annotation column added for annotators.
            ``None`` for task 2 (emotion labeling, handled separately).
    """

    num: int
    slug: str
    sid: str
    response_col_suffix: str
    quality_col: str | None = None


TASKS: Tuple[Task, ...] = (
    Task(1, "task1_empathetic_engagement", "P-R1", "R1", "EMPATHY_QUALITY"),
    Task(2, "task2_user_chatbot_emotion",  "P-R2", "R2", None),
    Task(3, "task3_question_quality",      "P-R3", "R3", "QUESTION_QUALITY"),
    Task(4, "task4_overall_dial",          "P-R",  "",   "OVERALL_QUALITY"),
)


# ---------------------------------------------------------------------------
# Sampling parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SamplingConfig:
    """Parameters for the per-annotator split described in
    ``human_annotations.txt``."""

    iaa_size: int = 7
    per_emotion_shared: int = 2
    per_emotion_task2_extra: int = 3
    sampling_random_state: int = 42
    shuffle_random_state: int = 4


SAMPLING = SamplingConfig()


# ---------------------------------------------------------------------------
# UID patterns
# ---------------------------------------------------------------------------

UID_PREFIX = "SFTANNO"


def task_uid_suffix(task_num: int) -> str:
    """UID suffix used for a given task (``-0001``…``-0004``)."""
    return f"-{task_num:04d}"
