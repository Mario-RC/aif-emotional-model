"""Shared utilities for the DPO Comparison Dataset candidate-curation pipeline.

Mirrors ``phase3-rlaif-alignment/reward-model/rm-comparison-dataset/candidate-curation/_lib.py`` (the data
naming differs but the logic is identical): the model registry, response-parsing
logic (emotion vs. utterance), embedding/similarity helpers, distinct-n metric
and the scoring formula used to rank candidate predictions.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = [
    "gemma-2-9b-it",
    "glm-4-9b-chat-1m",
    "Meta-Llama-3-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Phi-3-small-8k-instruct",
]
MODEL_NAMES = ["gemma2", "glm4", "llama3", "mistral", "phi3"]
MODEL_TO_NAME = dict(zip(MODELS, MODEL_NAMES))
REPO_ROOT = Path(__file__).resolve().parents[4]
CANONICAL_DATASETS = (
    REPO_ROOT / "phase2-sft-alignment/sft-model/sft-demonstration-dataset/data/sft_demonstration_dataset.json",
    REPO_ROOT / "phase2-sft-alignment/sft-model/sft-demonstration-dataset/data/sft_demonstration_dataset_test.json",
    REPO_ROOT / "phase3-rlaif-alignment/reward-model/rm-prompt-dataset/data/rm_prompt_dataset.json",
    REPO_ROOT / "phase3-rlaif-alignment/reward-model/rm-prompt-dataset/data/rm_prompt_dataset_test.json",
    REPO_ROOT / "phase3-rlaif-alignment/rlaif-model/ppo-unlabeled-prompts-dataset/data/ppo_unlabeled_prompts_dataset.json",
    REPO_ROOT / "phase3-rlaif-alignment/rlaif-model/ppo-unlabeled-prompts-dataset/data/ppo_unlabeled_prompts_dataset_test.json",
)


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def _prompt_value(row: dict | pd.Series) -> Any:
    return row.get("prompt") or row.get("input") or row.get("instruction")


def _target_value(row: dict | pd.Series) -> Any:
    return row.get("target") or row.get("output")


def canonical_metadata_by_content() -> dict[tuple[str, str], tuple[str, str]]:
    """Map ``(prompt, target)`` content to canonical ``(set, dialogue_id)``."""
    metadata: dict[tuple[str, str], tuple[str, str]] = {}
    for path in CANONICAL_DATASETS:
        for entry in read_json(path):
            key = (_clean_text(_prompt_value(entry)), _clean_text(_target_value(entry)))
            metadata[key] = (entry["set"], entry["dialogue_id"])
    return metadata


def add_canonical_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical ``dialogue_id``/``set`` columns and remove legacy ``did``."""
    metadata = canonical_metadata_by_content()
    dialogue_ids, source_sets = [], []
    missing: list[str] = []

    for _, row in df.iterrows():
        key = (_clean_text(_prompt_value(row)), _clean_text(_target_value(row)))
        values = metadata.get(key)
        if values is None:
            missing.append(_clean_text(_prompt_value(row))[:120])
            dialogue_ids.append(row.get("dialogue_id", ""))
            source_sets.append(row.get("set", ""))
            continue
        source_set, dialogue_id = values
        dialogue_ids.append(dialogue_id)
        source_sets.append(source_set)

    if missing:
        sample = "\n  - ".join(missing[:5])
        raise KeyError(f"Could not map {len(missing)} rows to canonical dialogue_id:\n  - {sample}")

    df = df.copy()
    df["dialogue_id"] = dialogue_ids
    df["set"] = source_sets
    return df.drop(columns=["did"], errors="ignore")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def split_emotions_in_response(predict_sft: str) -> tuple[list[str], list[str]]:
    r1_ini = predict_sft.find("(")
    r2_ini = predict_sft.find("(", r1_ini + 1) if r1_ini != -1 else -1
    r3_ini = predict_sft.find("(", r2_ini + 1) if r2_ini != -1 else -1

    r1_end = predict_sft.find(")")
    r2_end = predict_sft.find(")", r1_end + 1) if r1_end != -1 else -1
    r3_end = predict_sft.find(")", r2_end + 1) if r2_end != -1 else -1

    def slice_pair(start: int, end: int, next_start: int) -> tuple[str, str]:
        if start == -1 or end == -1:
            return "", ""
        utt = predict_sft[end + 1 : next_start if next_start != -1 else None].strip()
        emo = predict_sft[start + 1 : end].strip()
        return utt, emo

    utt1, emo1 = slice_pair(r1_ini, r1_end, r2_ini)
    utt2, emo2 = slice_pair(r2_ini, r2_end, r3_ini)
    utt3, emo3 = slice_pair(r3_ini, r3_end, -1)
    return [utt1, utt2, utt3], [emo1, emo2, emo3]


def split_emotions_for_columns(df: pd.DataFrame, columns: list[str]) -> list[list[list]]:
    out: list[list[list]] = []
    for col in columns:
        r_utts, r_emos = [], []
        for value in df[col]:
            utts, emos = split_emotions_in_response(value)
            r_utts.append(utts)
            r_emos.append(emos)
        out.append([r_utts, r_emos])
    return out


# ---------------------------------------------------------------------------
# Embeddings + semantic similarity
# ---------------------------------------------------------------------------

def get_embeddings(df: pd.DataFrame, split_columns: list[str], model) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for col in split_columns:
        utts = df[col]
        if isinstance(utts.iloc[0], str):
            import ast

            utts = utts.apply(ast.literal_eval)
        joined = utts.apply(lambda x: " ".join(x))
        out.append(model.encode(joined, task="text-matching"))
    return out


def pairwise_cosine_matrix(embeddings: list[np.ndarray]) -> tuple[list[list[float]], list[float]]:
    n = len(embeddings)
    matrix: list[list[float]] = []
    means: list[float] = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0.0 if i == j else float(embeddings[i] @ embeddings[j].T))
        matrix.append(row)
        means.append(float(np.mean(row)))
    return matrix, means


def per_row_semantic_similarity(
    df: pd.DataFrame, emb_columns: list[str]
) -> tuple[list[list[list[float]]], list[list[float]]]:
    matrices, means = [], []
    for idx in range(len(df)):
        embeddings = [np.array(df[col].iloc[idx]) for col in emb_columns]
        matrix, mean_vec = pairwise_cosine_matrix(embeddings)
        matrices.append(matrix)
        means.append(mean_vec)
    matrices_per_col = [[m[c] for m in matrices] for c in range(len(emb_columns))]
    means_per_col = [[m[c] for m in means] for c in range(len(emb_columns))]
    return matrices_per_col, means_per_col


# ---------------------------------------------------------------------------
# Distinct-n
# ---------------------------------------------------------------------------

def distinct_n(sentences: list[str], n: int) -> float:
    all_ngrams: list[tuple[str, ...]] = []
    total = 0
    for sentence in sentences:
        words = sentence.split()
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)
        total += len(ngrams)
    return len(set(all_ngrams)) / total if total > 0 else 0.0


def get_distinct_n_for_columns(df: pd.DataFrame, columns: list[str], n: int) -> list[list[float]]:
    out = []
    for col in columns:
        scores = []
        for idx in range(len(df)):
            scores.append(distinct_n([df["target"][idx], df[col][idx]], n))
        out.append(scores)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_equation(sts: float, d_n: float, length: float, alpha: float, beta: float, gamma: float) -> float:
    return alpha * (1 - sts) + beta * d_n + gamma * length


def rescale_distinct(x: float) -> float:
    return (x - 0.5) / 0.5


def length_score(reference_len: float, candidate_len: int, n: float = 8, m: float = 3) -> float:
    diff = abs(reference_len - candidate_len)
    t = diff / n
    return max(0.0, math.exp(-(t**m)))


def normalize_lengths(lengths: list[int], reference_idx_count: int = 1, n: float = 8, m: float = 3) -> list[float]:
    reference_len = float(np.mean(lengths[:reference_idx_count]))
    return [length_score(reference_len, candidate_len, n=n, m=m) for candidate_len in lengths[reference_idx_count:]]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def with_suffix(base: str, ext: str, is_test: bool) -> str:
    return f"{base}{'_test' if is_test else ''}.{ext}"
