"""Compute embeddings, semantic similarity, distinct-n and pick the chosen SFT prediction.

For each of the five SFT-trained models, this script:
1. Reads ``../llama-factory-predict/saves/<model>/emotional_balanced/dpo_comparison_dataset[_test]_results.json``
   and writes the per-model CSV under ``data/<model>/`` (and per-step intermediate
   files under the ``dpo_comparison_dataset_*`` prefix: ``*_splits.csv``, ``*_embs.csv``, ``*_distinct_n.csv``, ``*_predict_sft_chosen.csv``).
2. Splits each prediction into utterances and emotion tags.
3. Encodes utterances with ``jinaai/jina-embeddings-v3``.
4. Computes a pairwise cosine-similarity matrix and Distinct-2 / Distinct-3 scores.
5. Scores each candidate response with ``alpha (1-similarity) + beta * distinct + gamma * length_match``
   and stores the chosen prediction (best score) per row.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from sentence_transformers import SentenceTransformer

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
)

PREDICT_COLUMNS = [f"predict_sft_{i}" for i in range(8)]
ALL_RESPONSE_COLUMNS = ["target"] + PREDICT_COLUMNS
SPLIT_COLUMNS = [f"{c}_split" for c in ALL_RESPONSE_COLUMNS]
EMB_COLUMNS = [f"{c}_emb" for c in ALL_RESPONSE_COLUMNS]
SIM_COLUMNS = [f"{c}_semantic_similarity" for c in ALL_RESPONSE_COLUMNS]
SIM_MEAN_COLUMNS = [f"{c}_semantic_similarity_mean" for c in ALL_RESPONSE_COLUMNS]


def _llama_factory_path(model: str, is_test: bool) -> str:
    fname = "dpo_comparison_dataset_test_results.json" if is_test else "dpo_comparison_dataset_results.json"
    return f"../llama-factory-predict/saves/{model}/emotional_balanced/{fname}"


def _model_data_path(model: str, suffix: str, is_test: bool) -> str:
    short = MODEL_TO_NAME[model]
    base = f"data/{model}/dpo_comparison_dataset_{short}{suffix}"
    return with_suffix(base, "csv", is_test)


def json_to_csv(is_test: bool) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    for model in MODELS:
        df = pd.DataFrame(read_json(_llama_factory_path(model, is_test)))
        out_path = _model_data_path(model, "", is_test)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        dfs[model] = df
    return dfs


def add_split_columns(is_test: bool) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    for model in MODELS:
        df = pd.read_csv(_model_data_path(model, "", is_test), encoding="utf-8")
        splits = split_emotions_for_columns(df, ALL_RESPONSE_COLUMNS)
        for offset, split_col in enumerate(SPLIT_COLUMNS):
            df.insert(4 + offset * 2, split_col, splits[offset][0])
        df.to_csv(_model_data_path(model, "_splits", is_test), index=False)
        dfs[model] = df
    return dfs


def add_embedding_columns(is_test: bool, model_name: str = "jinaai/jina-embeddings-v3") -> dict[str, pd.DataFrame]:
    encoder = SentenceTransformer(model_name, trust_remote_code=True)
    dfs: dict[str, pd.DataFrame] = {}
    for model in MODELS:
        df = pd.read_csv(_model_data_path(model, "_splits", is_test), encoding="utf-8")
        embs = get_embeddings(df, SPLIT_COLUMNS, encoder)
        for offset, emb_col in enumerate(EMB_COLUMNS):
            df.insert(5 + offset * 3, emb_col, embs[offset].tolist())
        df.to_csv(_model_data_path(model, "_embs", is_test), index=False)
        dfs[model] = df
    return dfs


def add_similarity_and_distinct(is_test: bool) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    for model in MODELS:
        df = pd.read_csv(_model_data_path(model, "_embs", is_test), encoding="utf-8")
        for col in EMB_COLUMNS:
            df[col] = df[col].apply(eval)

        sim_per_col, sim_mean_per_col = per_row_semantic_similarity(df, EMB_COLUMNS)
        for offset, (sim_col, mean_col) in enumerate(zip(SIM_COLUMNS, SIM_MEAN_COLUMNS)):
            df.insert(6 + offset * 4, sim_col, sim_per_col[offset])
            df.insert(7 + offset * 4, mean_col, sim_mean_per_col[offset])

        candidate_cols = [f"predict_sft_{i}" for i in range(1, 8)]
        d2 = get_distinct_n_for_columns(df, candidate_cols, n=2)
        d3 = get_distinct_n_for_columns(df, candidate_cols, n=3)
        for offset, col in enumerate(candidate_cols):
            df[f"{col}_distinct_2"] = d2[offset]
            df[f"{col}_distinct_3"] = d3[offset]

        df.to_csv(_model_data_path(model, "_distinct_n", is_test), index=False)
        dfs[model] = df
    return dfs


def _candidate_score_for_row(df: pd.DataFrame, idx: int) -> tuple[list[float], int, str]:
    sim_target = df["target_semantic_similarity"].iloc[idx][2:6]
    sim_sft0 = df["predict_sft_0_semantic_similarity"].iloc[idx][2:6]
    semantic_similarity = [(t + s) / 2.0 for t, s in zip(sim_target, sim_sft0)]

    candidate_distinct = [
        rescale_distinct(df[f"predict_sft_{i}_distinct_2"].iloc[idx]) for i in range(1, 5)
    ]
    candidate_lengths = [
        len(df["target"].iloc[idx].split()),
        len(df["predict_sft_0"].iloc[idx].split()),
        *(len(df[f"predict_sft_{i}"].iloc[idx].split()) for i in range(1, 5)),
    ]
    length_scores = normalize_lengths(candidate_lengths, reference_idx_count=2, n=8, m=3)

    scores = [
        score_equation(sts, d_n, length, alpha=0.4, beta=0.4, gamma=0.2)
        for sts, d_n, length in zip(semantic_similarity, candidate_distinct, length_scores)
    ]
    chosen_idx = scores.index(max(scores)) + 1
    return scores, chosen_idx, df[f"predict_sft_{chosen_idx}"].iloc[idx]


def add_chosen_predictions(is_test: bool) -> None:
    import ast
    for model in MODELS:
        df = pd.read_csv(_model_data_path(model, "_distinct_n", is_test), encoding="utf-8")
        for col in SIM_COLUMNS:
            df[col] = df[col].apply(ast.literal_eval)

        scores, indices, chosen = [], [], []
        for idx in range(len(df)):
            s, i, c = _candidate_score_for_row(df, idx)
            scores.append(s)
            indices.append(i)
            chosen.append(c)

        df["predict_sft_score"] = scores
        df["predict_sft_chosen_index"] = indices
        df["predict_sft_chosen"] = chosen
        df.to_csv(_model_data_path(model, "_predict_sft_chosen", is_test), index=False)


def run_pipeline(is_test: bool = False) -> None:
    json_to_csv(is_test)
    add_split_columns(is_test)
    add_embedding_columns(is_test)
    add_similarity_and_distinct(is_test)
    add_chosen_predictions(is_test)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(is_test=args.test)
