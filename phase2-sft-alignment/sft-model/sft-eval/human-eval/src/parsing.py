"""Parse emotion-tagged dialogues and load SFT predictions.

Each dialogue sentence in the corpus is tagged as ``(EMOTION) utterance``.
The chatbot response is a sequence of three sentences (R1, R2, R3). This
module turns the raw JSON files produced by the Phase 2 SFT evaluation
into a single wide DataFrame with one row per test dialogue and one
column group per model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from .config import MODELS, SFT_RESULTS_FILENAME, SFT_SAVES_DIR, Model


EMOTION_TAG_RE = re.compile(r"\(([^)]*)\)")


# ---------------------------------------------------------------------------
# Dialogue parsing
# ---------------------------------------------------------------------------

def split_tagged_sentences(text: str, expected: int) -> List[tuple[str, str]]:
    """Split ``(EMO) utt. (EMO) utt. (EMO) utt.`` into tag / utterance pairs.

    The input mixes emotion tags and utterance text in a single string.
    We split on the ``(...)`` tags and pair each tag with the text that
    follows it, trimming whitespace.

    Args:
        text: A concatenation of tagged sentences.
        expected: Expected number of pairs (sanity-checked).

    Returns:
        A list of ``(emotion, utterance)`` tuples.
    """
    tags = EMOTION_TAG_RE.findall(text)
    parts = EMOTION_TAG_RE.split(text)
    # ``parts`` interleaves: [before_first_tag, tag1_content, between1, tag2_content, ...]
    # The regex capture group splits text into: pre-text, emotion, inter-text, emotion, ...
    # We only care about the inter-text segments.
    utterances = [seg.strip() for seg in parts[2::2]]
    if len(tags) != expected or len(utterances) != expected:
        raise ValueError(
            f"Expected {expected} tagged sentences, got "
            f"{len(tags)} tags and {len(utterances)} utterances in: {text!r}"
        )
    return list(zip(tags, utterances))


def parse_single_tagged(text: str) -> tuple[str, str]:
    """Parse a single ``(EMO) utt`` prompt."""
    tag_match = EMOTION_TAG_RE.search(text)
    if not tag_match:
        raise ValueError(f"No emotion tag found in: {text!r}")
    utterance = text[tag_match.end():].strip()
    return tag_match.group(1), utterance


# ---------------------------------------------------------------------------
# SFT results loading
# ---------------------------------------------------------------------------

def sft_results_path(model: Model, saves_dir: Path = SFT_SAVES_DIR) -> Path:
    return saves_dir / model.name / SFT_RESULTS_FILENAME


def load_sft_results_for_model(model: Model, saves_dir: Path = SFT_SAVES_DIR) -> pd.DataFrame:
    """Load one model's SFT test results and parse every dialogue.

    Each row of the returned DataFrame covers one test dialogue and
    contains: the user prompt (emotion + utterance), the three target
    sentences (emotion + utterance each), the three predicted sentences,
    and the raw predicted string.
    """
    path = sft_results_path(model, saves_dir)
    with path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    rows = []
    for idx, entry in enumerate(entries):
        prompt_emo, prompt_utt = parse_single_tagged(entry["input"])
        target_pairs = split_tagged_sentences(entry["target"], expected=3)
        predict_pairs = split_tagged_sentences(entry["predict_sft"], expected=3)

        row = {
            "UID":            f"SFTANNO-{idx:06d}",
            "SEG":            entry["input"],
            "TARGET":         entry["target"],
            "SFT_PREDICTION": entry["predict_sft"],
            "HUMAN_UTT":      prompt_utt,
            "HUMAN_EMO":      prompt_emo,
        }
        for i, ((tgt_emo, tgt_utt), (sft_emo, sft_utt)) in enumerate(
            zip(target_pairs, predict_pairs), start=1
        ):
            row[f"TARGET_R{i}"] = tgt_utt
            row[f"TARGET_EMO{i}"] = tgt_emo
            row[f"SFT_R{i}"] = sft_utt
            row[f"SFT_EMO{i}"] = sft_emo
        rows.append(row)

    return pd.DataFrame(rows)


def build_wide_dataframe(
    models: Sequence[Model] = MODELS,
    saves_dir: Path = SFT_SAVES_DIR,
) -> pd.DataFrame:
    """Build one wide DataFrame combining all models side-by-side.

    The wide DataFrame holds shared columns once (UID / SEG / TARGET /
    HUMAN_* / TARGET_*) and model-specific columns with a ``_<SHORT>``
    suffix (``SFT_PREDICTION_<SHORT>``, ``SFT_R{i}_<SHORT>``,
    ``SFT_EMO{i}_<SHORT>``).
    """
    shared_cols = [
        "UID", "SEG", "TARGET",
        "HUMAN_UTT", "HUMAN_EMO",
        "TARGET_R1", "TARGET_EMO1",
        "TARGET_R2", "TARGET_EMO2",
        "TARGET_R3", "TARGET_EMO3",
    ]
    model_specific_cols = ["SFT_PREDICTION", "SFT_R1", "SFT_EMO1",
                           "SFT_R2", "SFT_EMO2", "SFT_R3", "SFT_EMO3"]

    per_model = []
    for model in models:
        df = load_sft_results_for_model(model, saves_dir)
        renamed = df.rename(
            columns={c: f"{c}_{model.short}" for c in model_specific_cols}
        )
        per_model.append((model, renamed))

    # Shared columns come from the first model's DataFrame.
    base = per_model[0][1][shared_cols].copy()
    # Append all per-model columns, in the declared model order, grouped:
    # first every SFT_PREDICTION_*, then every R1 block, R2, R3.
    groups = ("SFT_PREDICTION",
              "SFT_R1", "SFT_EMO1",
              "SFT_R2", "SFT_EMO2",
              "SFT_R3", "SFT_EMO3")
    for group in groups:
        for model, df in per_model:
            col = f"{group}_{model.short}"
            base[col] = df[col].to_list()

    return base
