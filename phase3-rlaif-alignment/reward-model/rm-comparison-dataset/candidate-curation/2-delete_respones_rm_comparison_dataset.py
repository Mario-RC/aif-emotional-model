"""Combine the per-model chosen predictions, score and filter the worst per row.

Pipeline:
1. Read each ``data/<model>/rm_comparison_dataset_<short>_predict_sft_chosen.csv`` and
   write a tidy per-model JSON keeping only ``predict_sft_0`` and ``predict_sft_chosen``.
2. Combine the five per-model JSONs into ``data/rm_comparison_dataset_joined.json``.
3. Recompute embeddings, semantic similarity, distinct-2 and the score.
4. For each row drop the ``predict_sft_*`` column with the lowest score and write
   the filtered result to ``data/rm_comparison_dataset_filtered.json``.
"""

from __future__ import annotations

import argparse
import ast
import os
import random
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from _lib import (
    MODELS,
    MODEL_TO_NAME,
    get_distinct_n_for_columns,
    get_embeddings,
    normalize_lengths,
    per_row_semantic_similarity,
    read_json,
    rescale_distinct,
    score_equation,
    split_emotions_for_columns,
    with_suffix,
    write_json,
)

BASE_CHOSEN_COLUMNS = [
    "instruction", "history", "prompt", "target",
    "predict_sft_0", "predict_sft_chosen", "model",
]
IDENTITY_COLUMNS = ("dialogue_id", "set")
PREDICT_SFT_KEYS = [f"predict_sft_{m}_{stage}" for m in ["gemma2", "glm4", "llama3", "mistral", "phi3"] for stage in ("0", "x")]
PREDICT_SFT_COLUMNS = ["target"] + PREDICT_SFT_KEYS
SPLIT_COLUMNS = [f"{c}_split" for c in PREDICT_SFT_COLUMNS]
EMB_COLUMNS = [f"{c}_emb" for c in PREDICT_SFT_COLUMNS]


def _short(model: str) -> str:
    return MODEL_TO_NAME[model]


def _per_model_path(model: str, suffix: str, is_test: bool, ext: str = "csv") -> str:
    base = f"data/{model}/rm_comparison_dataset_{_short(model)}{suffix}"
    return with_suffix(base, ext, is_test)


def _chosen_columns(df: pd.DataFrame) -> list[str]:
    columns = [*BASE_CHOSEN_COLUMNS, *[col for col in IDENTITY_COLUMNS if col in df.columns]]
    if "dialogue_id" not in columns:
        raise KeyError("Prediction CSV is missing dialogue_id.")
    return columns


def _dialogue_id(entry: dict) -> str:
    dialogue_id = entry.get("dialogue_id")
    if not dialogue_id:
        raise KeyError("Prediction row is missing dialogue_id.")
    return dialogue_id


# ---------------------------------------------------------------------------
# Stage 1: per-model CSV → tidy JSON keeping only the predict_sft_chosen
# ---------------------------------------------------------------------------

def per_model_csv_to_json(is_test: bool) -> None:
    for model in tqdm(MODELS, desc="1/4 Per-model CSV to JSON", unit="model"):
        in_path = _per_model_path(model, "_predict_sft_chosen", is_test)
        df = pd.read_csv(in_path, encoding="utf-8")
        subset = df[_chosen_columns(df)].copy()
        subset["history"] = subset["history"].apply(ast.literal_eval)
        out_path = _per_model_path(model, "", is_test, ext="json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        write_json(subset.to_dict(orient="records"), out_path)


# ---------------------------------------------------------------------------
# Stage 2: combine per-model JSONs into a single ``rm_comparison_dataset_joined`` file
# ---------------------------------------------------------------------------

def combine_models(is_test: bool) -> None:
    per_model = {model: read_json(_per_model_path(model, "", is_test, ext="json")) for model in MODELS}
    base_model = MODELS[0]
    combined: list[dict] = []
    for idx in tqdm(range(len(per_model[base_model])), desc="2/4 Combine model rows", unit="row"):
        entry = {
            "instruction": per_model[base_model][idx]["instruction"],
            "history": per_model[base_model][idx]["history"],
            "prompt": per_model[base_model][idx]["prompt"],
            "target": per_model[base_model][idx]["target"],
        }
        for model in MODELS:
            short = _short(model)
            entry[f"predict_sft_{short}_0"] = per_model[model][idx]["predict_sft_0"]
            entry[f"predict_sft_{short}_x"] = per_model[model][idx]["predict_sft_chosen"]
        base_entry = per_model[base_model][idx]
        metadata = {"dialogue_id": _dialogue_id(base_entry)}
        if base_entry.get("set"):
            metadata["set"] = base_entry["set"]
        entry.update({
            "predict_sft_modified": "",
            "predict_sft_modified_label": "",
            "scores": [],
            **metadata,
        })
        combined.append(entry)

    write_json(combined, f"data/{with_suffix('rm_comparison_dataset_joined', 'json', is_test)}")


# ---------------------------------------------------------------------------
# Stage 3: print emotion-tag agreement for each candidate column
# ---------------------------------------------------------------------------

EMO_RE = re.compile(r"\(.*?\)")


def _safe_three_emotions(text: str) -> tuple[str, str, str]:
    found = EMO_RE.findall(text)
    found += [""] * (3 - len(found))
    return tuple(found[:3])


def report_emotion_agreement(is_test: bool, raise_on_mismatch: bool = False) -> None:
    combined = read_json(f"data/{with_suffix('rm_comparison_dataset_joined', 'json', is_test)}")
    keys = [k for k in PREDICT_SFT_KEYS if k.endswith(("_0", "_x"))]

    for key in tqdm(keys, desc="Report emotion agreement", unit="column"):
        user, chat, neutral = 0, 0, 0
        for entry in tqdm(combined, desc=key, unit="row", leave=False):
            t_user, t_chat, _ = _safe_three_emotions(entry["target"])
            p_user, p_chat, p_neutral = _safe_three_emotions(entry[key])
            user += p_user == t_user
            chat += p_chat == t_chat
            neutral += p_neutral == "(NEUTRAL)"
        n = len(combined)
        print(
            f"{key}\n"
            f"  user: {user/n*100:0.2f}%\n"
            f"  chatbot: {chat/n*100:0.2f}%\n"
            f"  neutral: {neutral/n*100:0.2f}%"
        )


# ---------------------------------------------------------------------------
# Stage 4: scoring + dropping the worst response per row
# ---------------------------------------------------------------------------

def add_scores(is_test: bool, model_name: str = "jinaai/jina-embeddings-v3") -> None:
    """Score each candidate response per row and persist the score back into the JSON."""
    in_json = f"data/{with_suffix('rm_comparison_dataset_joined', 'json', is_test)}"
    combined = read_json(in_json)
    df = pd.DataFrame(combined)

    splits = split_emotions_for_columns(df, PREDICT_SFT_COLUMNS, desc="3/4 Split scoring columns")
    for offset, split_col in enumerate(SPLIT_COLUMNS):
        df.insert(4 + offset * 2, split_col, splits[offset][0])

    encoder = SentenceTransformer(model_name, trust_remote_code=True)
    embs = get_embeddings(df, SPLIT_COLUMNS, encoder, desc="3/4 Embed scoring columns")
    for offset, emb_col in enumerate(EMB_COLUMNS):
        df.insert(5 + offset * 3, emb_col, embs[offset].tolist())

    sim_per_col, _ = per_row_semantic_similarity(df, EMB_COLUMNS, desc="3/4 Semantic rows")
    for offset, col in enumerate(PREDICT_SFT_COLUMNS):
        df.insert(6 + offset * 4, f"{col}_semantic_similarity", sim_per_col[offset])

    distinct_2 = get_distinct_n_for_columns(df, PREDICT_SFT_KEYS, n=2, desc="3/4 Distinct-2")
    for offset, key in enumerate(PREDICT_SFT_KEYS):
        df[f"{key}_distinct_2"] = distinct_2[offset]

    df["predict_sft_score"] = [
        _score_row(df, idx) for idx in tqdm(range(len(df)), desc="3/4 Score rows", unit="row")
    ]
    df.to_csv(f"data/{with_suffix('df_combined_data', 'csv', is_test)}", index=False)

    # Persist the per-row scores back into the JSON.
    for idx, score in tqdm(
        enumerate(df["predict_sft_score"]),
        total=len(df),
        desc="3/4 Persist scores",
        unit="row",
    ):
        combined[idx]["scores"] = score
    write_json(combined, in_json)


def _score_row(df: pd.DataFrame, idx: int) -> list[float]:
    target_sim = df["target_semantic_similarity"].iloc[idx][1:]
    candidate_distinct = [rescale_distinct(df[f"{k}_distinct_2"].iloc[idx]) for k in PREDICT_SFT_KEYS]
    candidate_lengths = [
        len(df["target"].iloc[idx].split()),
        *[len(df[k].iloc[idx].split()) for k in PREDICT_SFT_KEYS],
    ]
    length_scores = normalize_lengths(candidate_lengths, reference_idx_count=1, n=8, m=3)
    return [
        score_equation(sts, d_n, length, alpha=0.4, beta=0.4, gamma=0.2)
        for sts, d_n, length in zip(target_sim, candidate_distinct, length_scores)
    ]


def filter_worst_response(is_test: bool, n_drop: int = 3) -> None:
    """Drop the worst (lowest score) of the bottom-3 predictions for each row."""
    in_json = f"data/{with_suffix('rm_comparison_dataset_joined', 'json', is_test)}"
    combined = read_json(in_json)
    random.seed(42)

    for idx, entry in tqdm(enumerate(combined), total=len(combined), desc="4/4 Filter worst rows", unit="row"):
        scores = entry["scores"]
        bottom_indices = sorted(range(len(scores)), key=lambda i: scores[i])[:n_drop]
        random.seed(idx)
        drop_idx = random.choice(bottom_indices)
        entry.pop(PREDICT_SFT_KEYS[drop_idx], None)
        entry["scores"].pop(drop_idx)

    write_json(combined, f"data/{with_suffix('rm_comparison_dataset_filtered', 'json', is_test)}")


# ---------------------------------------------------------------------------
# Hyperparameter sweep (validation utility)
# ---------------------------------------------------------------------------

def validate_alpha_beta_gamma(is_test: bool) -> dict:
    """Sweep the (alpha, beta, gamma) trio and report mean / std / coefficient of variation."""
    df = pd.read_csv(f"data/{with_suffix('df_combined_data', 'csv', is_test)}")
    for col in [c for c in df.columns if c.endswith("semantic_similarity")]:
        df[col] = df[col].apply(ast.literal_eval)

    grid = {
        "alpha": [0.5, 0.5, 0.45, 0.45, 0.4],
        "beta":  [0.4, 0.3, 0.45, 0.35, 0.4],
        "gamma": [0.1, 0.2, 0.10, 0.20, 0.2],
    }
    results = []
    params = list(zip(grid["alpha"], grid["beta"], grid["gamma"]))
    for alpha, beta, gamma in tqdm(params, desc="Validate alpha/beta/gamma", unit="combo"):
        scores: list[float] = []
        for idx in tqdm(range(len(df)), desc=f"a={alpha} b={beta} g={gamma}", unit="row", leave=False):
            target_sim = df["target_semantic_similarity"].iloc[idx][1:]
            candidate_distinct = [rescale_distinct(df[f"{k}_distinct_2"].iloc[idx]) for k in PREDICT_SFT_KEYS]
            lengths = [
                len(df["target"].iloc[idx].split()),
                *[len(df[k].iloc[idx].split()) for k in PREDICT_SFT_KEYS],
            ]
            length_scores = normalize_lengths(lengths, reference_idx_count=1, n=8, m=3)
            for sts, d_n, length in zip(target_sim, candidate_distinct, length_scores):
                scores.append(score_equation(sts, d_n, length, alpha, beta, gamma))
        mean_s = float(np.mean(scores))
        std_s = float(np.std(scores))
        results.append({
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "mean_score": mean_s, "std_score": std_s,
            "cv_score": std_s / mean_s if mean_s else 0.0,
        })

    best_mean = max(results, key=lambda r: r["mean_score"])
    best_cv = min(results, key=lambda r: r["cv_score"])
    print("Best by mean:", best_mean)
    print("Best by CV  :", best_cv)
    return {"best_mean": best_mean, "best_cv": best_cv}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline(is_test: bool = False) -> None:
    per_model_csv_to_json(is_test)
    combine_models(is_test)
    add_scores(is_test)
    filter_worst_response(is_test)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--report-emotions", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(is_test=args.test)
    if args.report_emotions:
        report_emotion_agreement(is_test=args.test)
    if args.validate:
        validate_alpha_beta_gamma(is_test=args.test)
