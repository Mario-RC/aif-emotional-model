"""Build the XLSX files annotators fill in to evaluate SFT / PPO / DPO responses.

Pipeline:
1. Load ``demonstration_prompt_rlaif_data_test_results.json`` for the three best
   models (one per RL technique) and assemble a wide dataframe with one row per
   dialogue holding raw + parsed user turns and per-model responses.
2. Split each turn / response into utterance + emotion-tag columns.
3. Build Task 1 (rank 3 responses) and Task 2 (binary adequacy) datasets keeping
   a balanced distribution of TARGET_EMO2 emotions.
4. Write per-task XLSX files: ``data/task1.xlsx`` and ``data/task2.xlsx``.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

DEFAULT_BEST_MODELS = [
    ("gemma-2-9b-it", "predict_sft_dpr_1_4"),
    ("gemma-2-9b-it", "predict_ppo_1_4"),
    ("gemma-2-9b-it", "predict_dpo_3"),
]
DEFAULT_DATA_TEST_DF = "data/data_test_df.csv"
DEFAULT_TASK1_AUX = "data/task1_aux.csv"
DEFAULT_TASK2_AUX = "data/task2_aux.csv"
DEFAULT_TASK1_XLSX = "data/task1.xlsx"
DEFAULT_TASK2_XLSX = "data/task2.xlsx"

LEVEL_TO_LABEL = {5: "Very High", 4: "High", 3: "Medium", 2: "Low", 1: "Very Low"}


# ---------------------------------------------------------------------------
# Stage 1: build the wide dataframe
# ---------------------------------------------------------------------------

def _load_best_model_predictions(best_models: list[tuple[str, str]]) -> tuple[list[dict], list[dict], list[dict]]:
    """Read the SFT / PPO / DPO JSONs for the three best (model, predict_field) pairs."""
    paths = [
        f"../../rlaif-llama-factory-training/saves/{model}/emotional_balanced/demonstration_prompt_rlaif_data_test_results.json"
        for model, _ in best_models
    ]
    out: list[list[dict]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            out.append(json.loads(f.read()))
    return out[0], out[1], out[2]


def _split_emo_user(values: Iterable[str]) -> tuple[list[str], list[str]]:
    user_utt, user_emo = [], []
    for user_t in values:
        emo_ini = user_t.find("(")
        emo_end = user_t.find(")")
        user_utt.append(user_t[emo_end + 2 :].strip())
        user_emo.append(user_t[emo_ini : emo_end + 1].strip().replace("(", "").replace(")", ""))
    return user_utt, user_emo


@dataclass
class ChatbotSplit:
    r1_utt: list[str]
    r1_emo: list[str]
    r2_utt: list[str]
    r2_emo: list[str]
    r3_utt: list[str]
    r3_emo: list[str]


def _split_emo_chatbot(values: Iterable[str]) -> ChatbotSplit:
    r1_utt, r2_utt, r3_utt = [], [], []
    r1_emo, r2_emo, r3_emo = [], [], []
    for response in values:
        ini = [i for i in range(len(response)) if response.startswith("(", i)]
        end = [i for i in range(len(response)) if response.startswith(")", i)]
        r1_utt.append(response[end[0] + 2 : ini[1]].strip())
        r1_emo.append(response[ini[0] : end[0] + 1].strip().replace("(", "").replace(")", ""))
        r2_utt.append(response[end[1] + 2 : ini[2]].strip())
        r2_emo.append(response[ini[1] : end[1] + 1].strip().replace("(", "").replace(")", ""))
        r3_utt.append(response[end[2] + 2 :].lstrip())
        r3_emo.append(response[ini[2] : end[2] + 1].strip().replace("(", "").replace(")", ""))
    return ChatbotSplit(r1_utt, r1_emo, r2_utt, r2_emo, r3_utt, r3_emo)


def _wrap_chatbot_test(split: ChatbotSplit) -> list[str]:
    """Build the ``(EMPATHY) r1 (EMO2) r2 (EMO3) r3`` strings used by annotators."""
    return [
        f"(EMPATHY) {a} ({b}) {c} ({d}) {e}"
        for a, b, c, d, e in zip(split.r1_utt, split.r2_emo, split.r2_utt, split.r3_emo, split.r3_utt)
    ]


def _join_chatbot_response(split: ChatbotSplit) -> list[str]:
    return [f"{a} {b} {c}" for a, b, c in zip(split.r1_utt, split.r2_utt, split.r3_utt)]


def _add_user_turn_columns(df: pd.DataFrame, prefix: str, source_column: str, insert_at: int) -> int:
    utt, emo = _split_emo_user(df[source_column])
    df.insert(insert_at, f"{prefix}_R", utt)
    df.insert(insert_at + 1, f"USER_{prefix.split('_')[-1]}", emo)
    return insert_at + 2


def _add_chatbot_turn_columns(
    df: pd.DataFrame, prefix: str, source_column: str, insert_at: int
) -> int:
    split = _split_emo_chatbot(df[source_column])
    df.insert(insert_at, f"{prefix}_TEST", _wrap_chatbot_test(split))
    df.insert(insert_at + 1, f"{prefix}_R", _join_chatbot_response(split))
    df.insert(insert_at + 2, f"{prefix}_EMO1", split.r1_emo)
    df.insert(insert_at + 3, f"{prefix}_R1", split.r1_utt)
    df.insert(insert_at + 4, f"{prefix}_EMO2", split.r2_emo)
    df.insert(insert_at + 5, f"{prefix}_R2", split.r2_utt)
    df.insert(insert_at + 6, f"{prefix}_EMO3", split.r3_emo)
    df.insert(insert_at + 7, f"{prefix}_R3", split.r3_utt)
    return insert_at + 8


def build_wide_dataframe(best_models: list[tuple[str, str]] = DEFAULT_BEST_MODELS) -> pd.DataFrame:
    sft, ppo, dpo = _load_best_model_predictions(best_models)
    rows = {key: [] for key in [
        "DID", "USER_T1", "CHATBOT_T1", "USER_T2", "CHATBOT_T2",
        "USER_T3", "CHATBOT_T3", "USER_T4", "TARGET", "SFT", "PPO", "DPO",
    ]}
    for dial in dpo:
        rows["DID"].append(dial["did"])
        rows["USER_T1"].append(dial["history"][0][0])
        rows["CHATBOT_T1"].append(dial["history"][0][1])
        rows["USER_T2"].append(dial["history"][1][0])
        rows["CHATBOT_T2"].append(dial["history"][1][1])
        rows["USER_T3"].append(dial["history"][2][0])
        rows["CHATBOT_T3"].append(dial["history"][2][1])
        rows["USER_T4"].append(dial["prompt"])
        rows["TARGET"].append(dial["target"])
    rows["SFT"] = [dial[best_models[0][1]] for dial in sft]
    rows["PPO"] = [dial[best_models[1][1]] for dial in ppo]
    rows["DPO"] = [dial[best_models[2][1]] for dial in dpo]
    return pd.DataFrame(rows)


def add_split_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-turn (USER_*) and per-response ((CHATBOT|TARGET|SFT|PPO|DPO)_*) columns."""
    insert_at = 2
    insert_at = _add_user_turn_columns(df, prefix="USER_T1", source_column="USER_T1", insert_at=insert_at)
    insert_at += 1  # skip CHATBOT_T1 column already in the base df
    insert_at = _add_chatbot_turn_columns(df, prefix="CHATBOT_T1", source_column="CHATBOT_T1", insert_at=insert_at)

    insert_at = _add_user_turn_columns(df, prefix="USER_T2", source_column="USER_T2", insert_at=insert_at)
    insert_at += 1
    insert_at = _add_chatbot_turn_columns(df, prefix="CHATBOT_T2", source_column="CHATBOT_T2", insert_at=insert_at)

    insert_at = _add_user_turn_columns(df, prefix="USER_T3", source_column="USER_T3", insert_at=insert_at)
    insert_at += 1
    insert_at = _add_chatbot_turn_columns(df, prefix="CHATBOT_T3", source_column="CHATBOT_T3", insert_at=insert_at)

    insert_at = _add_user_turn_columns(df, prefix="USER_T4", source_column="USER_T4", insert_at=insert_at)
    insert_at += 1
    for prefix in ("TARGET", "SFT", "PPO", "DPO"):
        insert_at = _add_chatbot_turn_columns(df, prefix=prefix, source_column=prefix, insert_at=insert_at)
    return df


# ---------------------------------------------------------------------------
# Stage 2: split into Task 1 / Task 2 with balanced TARGET_EMO2
# ---------------------------------------------------------------------------

def split_task1_task2(df: pd.DataFrame, task1_per_emotion: int = 22, task2_per_emotion: int = 18) -> tuple[pd.DataFrame, pd.DataFrame]:
    task1 = df.groupby("TARGET_EMO2").head(task1_per_emotion).drop_duplicates()
    task2 = (
        df[~df["DID"].isin(task1["DID"].tolist())]
        .groupby("TARGET_EMO2")
        .head(task2_per_emotion)
        .drop_duplicates()
    )
    return task1, task2


# ---------------------------------------------------------------------------
# Stage 3: render Task 1 and Task 2 XLSX
# ---------------------------------------------------------------------------

def write_task1_xlsx(task1: pd.DataFrame, out_path: str = DEFAULT_TASK1_XLSX) -> None:
    columns = ["DID", "USER_T2", "CHATBOT_T2_TEST", "USER_T3", "CHATBOT_T3_TEST", "USER_T4",
               "SFT_TEST", "PPO_TEST", "DPO_TEST"]
    df = task1[columns].rename(columns={
        "USER_T2": "USER_T1", "CHATBOT_T2_TEST": "CHATBOT_T1",
        "USER_T3": "USER_T2", "CHATBOT_T3_TEST": "CHATBOT_T2",
        "USER_T4": "USER_T3",
        "SFT_TEST": "RESPONSE_1", "PPO_TEST": "RESPONSE_2", "DPO_TEST": "RESPONSE_3",
    })
    df.insert(9, "RESPONSE_1_RANK", "")
    df.insert(10, "RESPONSE_2_RANK", "")
    df.insert(11, "RESPONSE_3_RANK", "")
    df["COMMENTS"] = ""
    df.to_excel(out_path, index=False)


def _load_did_expression_level() -> pd.DataFrame:
    rlaif = pd.read_csv("data/rlaif_data_models_results_test.csv", usecols=["DID", "EXPRESSION_LEVEL"])
    comparison = pd.read_csv("data/comparison_data_models_results_test.csv", usecols=["DID", "EXPRESSION_LEVEL"])
    df = pd.concat([rlaif, comparison])
    df["EXPRESSION_LEVEL"] = df["EXPRESSION_LEVEL"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    return df


def _balance_by_expression_level(emotion_rows: pd.DataFrame, rows_per_group: int) -> pd.DataFrame:
    out = pd.DataFrame()
    for level in (5, 4, 3, 2, 1):
        out = pd.concat([
            out,
            emotion_rows[emotion_rows["EXPRESSION_LEVEL"].apply(lambda x: x[1] == level)].head(rows_per_group // 5),
        ])
    return out


def _build_task2_grouped(task2: pd.DataFrame, did_expression_level: pd.DataFrame) -> pd.DataFrame:
    task2 = task2.copy()
    task2["EXPRESSION_LEVEL"] = task2["DID"].map(
        did_expression_level.set_index("DID")["EXPRESSION_LEVEL"]
    )

    emotions = task2["TARGET_EMO2"].unique()
    rows_per_group = task2["TARGET_EMO2"].value_counts().iloc[0] // 3

    grouped: list[pd.DataFrame] = []
    for emotion in emotions:
        emotion_rows = task2[task2["TARGET_EMO2"] == emotion].sample(frac=1, random_state=42)
        sft_rows = _balance_by_expression_level(emotion_rows, rows_per_group)
        emotion_rows = emotion_rows.drop(sft_rows.index)
        ppo_rows = _balance_by_expression_level(emotion_rows, rows_per_group)
        emotion_rows = emotion_rows.drop(ppo_rows.index)
        dpo_rows = _balance_by_expression_level(emotion_rows, rows_per_group)

        sft_rows["RESPONSE_TYPE"] = "SFT_TEST"
        ppo_rows["RESPONSE_TYPE"] = "PPO_TEST"
        dpo_rows["RESPONSE_TYPE"] = "DPO_TEST"
        grouped.extend([sft_rows, ppo_rows, dpo_rows])

    return pd.concat(grouped).reset_index(drop=True)


def _map_expression_levels_to_labels(did_expression_level: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric levels to ``Very High``/``High``/...``Very Low`` + dimension suffix."""
    def to_labels(expr):
        if isinstance(expr, str):
            try:
                expr = ast.literal_eval(expr)
            except Exception:
                return expr
        return [LEVEL_TO_LABEL.get(num, num) for num in expr]

    df = did_expression_level.copy()
    df["EXPRESSION_LEVEL"] = df["EXPRESSION_LEVEL"].apply(to_labels)
    suffixes = (" Empathy", " Emotion", " Engagement")
    df["EXPRESSION_LEVEL"] = df.apply(
        lambda row: [
            f"{label}{suffixes[i]}" if i < 3 else label
            for i, label in enumerate(row["EXPRESSION_LEVEL"])
        ],
        axis=1,
    )
    return df


def write_task2_xlsx(task2: pd.DataFrame, out_path: str = DEFAULT_TASK2_XLSX) -> None:
    did_expression_level = _load_did_expression_level()
    task2_grouped = _build_task2_grouped(task2, did_expression_level)

    task2_grouped["RESPONSE"] = task2_grouped.apply(
        lambda row: row[row["RESPONSE_TYPE"]] if row["RESPONSE_TYPE"] in ("SFT_TEST", "PPO_TEST", "DPO_TEST") else row["RESPONSE_TYPE"],
        axis=1,
    )

    keep_columns = ["DID", "USER_T2", "CHATBOT_T2_TEST", "USER_T3", "CHATBOT_T3_TEST", "USER_T4",
                    "RESPONSE", "TARGET_EMO2"]
    df = task2_grouped[keep_columns].rename(columns={
        "USER_T2": "USER_T1", "CHATBOT_T2_TEST": "CHATBOT_T1",
        "USER_T3": "USER_T2", "CHATBOT_T3_TEST": "CHATBOT_T2",
        "USER_T4": "USER_T3",
        "TARGET_EMO2": "RESPONSE_EMOTION",
    })
    df["EMPATHY_ADEQUACY"] = ""
    df["EMOTION_ADEQUACY"] = ""
    df["ENGAGEMENT_ADEQUACY"] = ""
    df["COMMENTS"] = ""

    expression_with_labels = _map_expression_levels_to_labels(did_expression_level)
    df["EXPRESSION_LEVEL"] = df["DID"].map(expression_with_labels.set_index("DID")["EXPRESSION_LEVEL"])
    df["EXPRESSION_LEVEL"] = df.apply(
        lambda row: [
            f"{' '.join(level.split()[:-1])} {row['RESPONSE_EMOTION'].capitalize()} {level.split()[-1]}" if "Emotion" in level else level
            for level in row["EXPRESSION_LEVEL"]
        ],
        axis=1,
    )
    df = df.drop(columns=["RESPONSE_EMOTION"])
    df.to_excel(out_path, index=False)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_human_annotation_files(
    best_models: list[tuple[str, str]] = DEFAULT_BEST_MODELS,
    data_test_df: str = DEFAULT_DATA_TEST_DF,
    task1_aux: str = DEFAULT_TASK1_AUX,
    task2_aux: str = DEFAULT_TASK2_AUX,
    task1_xlsx: str = DEFAULT_TASK1_XLSX,
    task2_xlsx: str = DEFAULT_TASK2_XLSX,
) -> None:
    df = build_wide_dataframe(best_models)
    df = add_split_columns(df)
    df.to_csv(data_test_df, index=False, encoding="utf-8")

    task1, task2 = split_task1_task2(df)
    task1.to_csv(task1_aux, index=False, encoding="utf-8")
    task2.to_csv(task2_aux, index=False, encoding="utf-8")

    write_task1_xlsx(task1, task1_xlsx)
    write_task2_xlsx(task2, task2_xlsx)


def _parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


if __name__ == "__main__":
    _parse_args()
    generate_human_annotation_files()
