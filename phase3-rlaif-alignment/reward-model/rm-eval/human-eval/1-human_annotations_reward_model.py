"""Build a human-annotation reward-model dataset from per-annotator XLSX files.

Pipeline:
1. Load the SFT predictions for each model from
   ``../../../../phase2-sft-alignment/sft-model/sft-llama-factory-legacy/saves/<model>/emotional_balanced/
   demonstration_data_emotional_balanced_test_results.json`` and concatenate
   them into a single dataframe keyed by UID.
2. Load each annotator's XLSX files (task1 = empathy, task3 = question, task4 =
   overall), concatenate them and join on the source ``USER_PROMPT`` to recover
   the DID / SYSTEM / HISTORY / TARGET.
3. Convert the per-task ranks into scores, sum across tasks, then convert the
   summed scores back into ranks.
4. From those ranks build pairwise winner/loser comparisons and join the
   chosen / rejected SFT responses.
5. Save the LLaMA-Factory-style reward-model JSON under
   ``results/human_annotations_reward_model_test.json``.
"""

from __future__ import annotations

import argparse
import ast
import json
from itertools import combinations

import pandas as pd
from scipy.stats import rankdata

MODELS = [
    "glm-4-9b-chat-1m",
    "gemma-2-9b-it",
    "Meta-Llama-3-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Phi-3-small-8k-instruct",
]

ANNOTATORS = ["kallirroi", "marcos", "jaime", "luisfernando", "mario"]
PHASE2_RESULTS_ROOT = "../../../../phase2-sft-alignment/sft-model/sft-llama-factory-legacy/saves"

TASKS = {
    "t1": "task1_empathetic_engagement",
    "t3": "task3_question_quality",
    "t4": "task4_overall_dial",
}
TASK_VALUE_COLUMNS = {
    "t1": [f"MODEL{i}_EMPATHY_QUALITY" for i in range(1, 6)],
    "t3": [f"MODEL{i}_QUESTION_QUALITY" for i in range(1, 6)],
    "t4": [f"MODEL{i}_OVERALL_QUALITY" for i in range(1, 6)],
}


# ---------------------------------------------------------------------------
# Step 1: load SFT predictions per model
# ---------------------------------------------------------------------------

def load_sft_predictions() -> pd.DataFrame:
    rows: list[dict] = []
    counter = -1
    for model_name in MODELS:
        path = (
            f"{PHASE2_RESULTS_ROOT}/{model_name}/emotional_balanced/"
            "demonstration_data_emotional_balanced_test_results.json"
        )
        with open(path, "r", encoding="utf-8") as f:
            for dial in json.load(f):
                counter += 1
                rows.append({
                    "UID": f"SFTANNO-{counter:06d}",
                    "MODEL": model_name,
                    "SEG": dial["input"],
                    "TARGET": dial["target"],
                    "SFT_PREDICTION": dial["predict_sft"],
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2: load and concatenate annotator XLSX files
# ---------------------------------------------------------------------------

def _load_one_task_per_annotator(task_key: str) -> pd.DataFrame:
    task_suffix = TASKS[task_key]
    frames = [
        pd.read_excel(f"results/{annotator}/anno{idx + 1}_{task_suffix}.xlsx")
        for idx, annotator in enumerate(ANNOTATORS)
    ]
    df = pd.concat(frames, ignore_index=True)
    return df.drop(columns=["COMMENTS"])


def _augment_with_metadata(annotations: pd.DataFrame, prompt_to_meta: dict) -> pd.DataFrame:
    matches = annotations["USER_PROMPT"].apply(
        lambda prompt: [
            [instruction, prompt_to_meta[instruction]]
            for instruction in prompt_to_meta if prompt in instruction
        ]
    )
    annotations = annotations.copy()
    annotations["DID"] = [item[1][0] for sub in matches for item in sub]
    annotations["SYSTEM"] = [item[1][1] for sub in matches for item in sub]
    annotations["HISTORY"] = [item[1][2] for sub in matches for item in sub]
    annotations["PROMPT"] = [item[0] for sub in matches for item in sub]
    annotations["TARGET"] = [item[1][3] for sub in matches for item in sub]
    return annotations.drop_duplicates(subset=["DID"], keep=False).drop(columns=["USER_PROMPT"])


def load_annotations(prompt_to_meta: dict) -> dict[str, pd.DataFrame]:
    return {
        task_key: _augment_with_metadata(_load_one_task_per_annotator(task_key), prompt_to_meta)
        for task_key in TASKS
    }


# ---------------------------------------------------------------------------
# Step 3: attach the per-model SFT predictions to each task dataframe
# ---------------------------------------------------------------------------

def attach_sft_predictions(task_dfs: dict[str, pd.DataFrame], test_results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    seg_to_predictions = dict(zip(
        test_results["SEG"],
        zip(
            test_results["SFT_PREDICTION_GLM4"],
            test_results["SFT_PREDICTION_GEMMA"],
            test_results["SFT_PREDICTION_LLAMA3"],
            test_results["SFT_PREDICTION_MISTRAL"],
            test_results["SFT_PREDICTION_PHI3"],
        ),
    ))

    for df in task_dfs.values():
        matches = df["PROMPT"].apply(
            lambda prompt: [seg_to_predictions[s] for s in seg_to_predictions if prompt in s]
        )
        for model_idx in range(5):
            df[f"MODEL{model_idx + 1}_RESPONSE"] = [
                tup[model_idx] for sub in matches for tup in sub
            ]
    return task_dfs


# ---------------------------------------------------------------------------
# Step 4: convert ranks → scores, sum across tasks, then re-rank
# ---------------------------------------------------------------------------

def _rank_to_score(rank: int) -> int:
    return 6 - rank


def aggregate_task_scores(task_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    t1_models = [list(map(_rank_to_score, row)) for row in task_dfs["t1"][TASK_VALUE_COLUMNS["t1"]].itertuples(index=False)]
    t3_models = [list(map(_rank_to_score, row)) for row in task_dfs["t3"][TASK_VALUE_COLUMNS["t3"]].itertuples(index=False)]
    t4_models = [list(row) for row in task_dfs["t4"][TASK_VALUE_COLUMNS["t4"]].itertuples(index=False)]

    ranks_df = pd.DataFrame({
        "DID": task_dfs["t4"]["DID"].values,
        "t1_scores": t1_models,
        "t3_scores": t3_models,
        "t4_scores": t4_models,
    })
    summed = ranks_df.groupby("DID").sum()
    summed_per_did = summed.apply(
        lambda row: [sum(x) for x in zip(row["t1_scores"], row["t3_scores"], row["t4_scores"])], axis=1
    ).tolist()

    tx = task_dfs["t4"].copy().drop(columns=TASK_VALUE_COLUMNS["t4"])
    tx["RATE_OVERALL"] = summed_per_did
    tx["RANK"] = tx["RATE_OVERALL"].apply(lambda scores: rankdata(scores, method="dense").tolist())
    return tx


# ---------------------------------------------------------------------------
# Step 5: pairwise comparisons + winner/loser response lookup
# ---------------------------------------------------------------------------

def _create_pairwise_comparisons(ranking: list[int]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i, j in combinations(range(len(ranking)), 2):
        if ranking[i] == ranking[j]:
            continue
        pairs.append((i + 1, j + 1) if ranking[i] < ranking[j] else (j + 1, i + 1))
    return pairs


def build_pairwise_dataframe(tx: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in tx.iterrows():
        for winner, loser in _create_pairwise_comparisons(row["RANK"]):
            rows.append({"DID": row["DID"], "WINNER": winner, "LOSER": loser})
    df = pd.DataFrame(rows)

    enriched: list[dict] = []
    for _, row in df.iterrows():
        record = tx.loc[tx["DID"] == row["DID"]].iloc[0]
        enriched.append({
            "DID": row["DID"],
            "WINNER": row["WINNER"],
            "LOSER": row["LOSER"],
            "SYSTEM": record["SYSTEM"],
            "HISTORY": record["HISTORY"],
            "PROMPT": record["PROMPT"],
            "WINNER_RESPONSE": record[f"MODEL{row['WINNER']}_RESPONSE"],
            "LOSER_RESPONSE": record[f"MODEL{row['LOSER']}_RESPONSE"],
        })
    out = pd.DataFrame(enriched)

    out.insert(0, "UID", out.groupby("DID").cumcount())
    out["UID"] = out["DID"] + "-" + out["UID"].astype(str).str.zfill(4)
    out["UID"] = out["UID"].str.replace("GPT4", "COMPAR")
    out["DID"] = out["DID"].str.replace("GPT4", "COMPAR")
    return out.drop(columns=["DID"])


# ---------------------------------------------------------------------------
# Step 6: format as LLaMA-Factory reward-model JSON
# ---------------------------------------------------------------------------

def to_reward_model_format(comparison_df: pd.DataFrame) -> list[dict]:
    entries: list[dict] = []
    for uid in comparison_df["UID"].unique():
        df_uid = comparison_df[comparison_df["UID"] == uid].reset_index(drop=True)
        system = df_uid["SYSTEM"].values[0]
        history = df_uid["HISTORY"].values[0]
        prompt = df_uid["PROMPT"].values[0]
        winner_response = df_uid["WINNER_RESPONSE"].values[0]
        loser_response = df_uid["LOSER_RESPONSE"].values[0]
        entries.append({
            "conversations": [
                {"from": "system", "value": system},
                {"from": "human", "value": history[0][0]},
                {"from": "gpt", "value": history[0][1]},
                {"from": "human", "value": history[1][0]},
                {"from": "gpt", "value": history[1][1]},
                {"from": "human", "value": history[2][0]},
                {"from": "gpt", "value": history[2][1]},
                {"from": "human", "value": prompt},
            ],
            "chosen": {"from": "gpt", "value": winner_response},
            "rejected": {"from": "gpt", "value": loser_response},
            "uid": uid,
        })
    return entries


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run() -> None:
    test_results = pd.read_csv("results/demonstration_data_emotional_balanced_test_results_dial.csv")
    load_sft_predictions()  # kept for inspection / regeneration

    train_dev = pd.read_json("../../../rlaif-model/ppo-unlabeled-prompts-dataset/data/ppo_unlabeled_prompts_dataset.json")
    test = pd.read_json("../../../rlaif-model/ppo-unlabeled-prompts-dataset/data/ppo_unlabeled_prompts_dataset_test.json")
    full = pd.concat([train_dev, test], ignore_index=True)
    prompt_to_meta = dict(zip(
        full["instruction"],
        zip(full["did"], full["system"], full["history"], full["output"]),
    ))

    task_dfs = load_annotations(prompt_to_meta)
    task_dfs = attach_sft_predictions(task_dfs, test_results)

    tx = aggregate_task_scores(task_dfs)
    tx.to_csv("results/tx_anno_all.csv", index=False)

    comparison_df = build_pairwise_dataframe(tx)
    comparison_df.to_csv("results/comparison_data_df.csv", index=False)

    comparison_json = ast.literal_eval(comparison_df.to_json(orient="records"))
    with open("results/human_annotations_response_test.json", "w", encoding="utf-8") as f:
        json.dump(comparison_json, f, ensure_ascii=False, indent=2)

    reward_model_entries = to_reward_model_format(
        pd.read_json("results/human_annotations_response_test.json")
    )
    with open("results/human_annotations_reward_model_test.json", "w", encoding="utf-8") as f:
        json.dump(reward_model_entries, f, ensure_ascii=False, indent=2)


def _parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


if __name__ == "__main__":
    _parse_args()
    run()
