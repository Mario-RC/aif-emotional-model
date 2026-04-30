"""Split the per-turn RLAIF data into emotion-balanced train/test JSON files.

Test split: 2 samples per (EMOTION_RESPONSE_1, EMOTION_RESPONSE_2) pair, taken
from the last turn of each dialogue (UID ending in "6"); the remaining dialogues
form the train_dev split. Both splits are exported as LLaMA-Factory-style JSON.

Outputs:
    data/ppo_unlabeled_prompts_dataset_original.json
    data/ppo_unlabeled_prompts_dataset_test_original.json
"""

from __future__ import annotations

import argparse

import pandas as pd

from _lib import build_dialogue_record, write_json

DEFAULT_TURNS = "data/ppo_unlabeled_prompts_dataset_turns.csv"
DEFAULT_TRAIN = "data/ppo_unlabeled_prompts_dataset_original.json"
DEFAULT_TEST = "data/ppo_unlabeled_prompts_dataset_test_original.json"
N_TIMES_PER_PAIR = 2


def _select_test_indices(df_turns: pd.DataFrame) -> list[int]:
    """Sample N_TIMES_PER_PAIR per emotion pair (E_R1 x E_R2) and expand to full dialogues."""
    df_responses = df_turns[df_turns["UID"].str.endswith("6")]
    emotions = list(df_turns["EMOTION_RESPONSE_2"].unique())

    pair_indices: list[pd.Index] = []
    for e1 in emotions:
        for e2 in emotions:
            pair_df = df_responses[
                (df_responses["EMOTION_RESPONSE_1"] == e1)
                & (df_responses["EMOTION_RESPONSE_2"] == e2)
            ]
            pair_indices.append(pair_df.sample(n=N_TIMES_PER_PAIR).index)

    test_set: list[int] = []
    for index_list in pair_indices:
        for index in index_list:
            test_set.extend(range(index - 3, index + 1))
    test_set.sort()
    return test_set


def _df_to_records(df_split: pd.DataFrame) -> list[dict]:
    """Group by DID and convert each dialogue to a single training record."""
    records = []
    for did in df_split["DID"].unique():
        df_did = df_split[df_split["DID"] == did].reset_index(drop=True)
        records.append(build_dialogue_record(df_did, did))
    return records


def split_train_test(
    turns_csv: str = DEFAULT_TURNS,
    train_json: str = DEFAULT_TRAIN,
    test_json: str = DEFAULT_TEST,
) -> None:
    df_turns = pd.read_csv(turns_csv, encoding="utf-8")

    test_idx = _select_test_indices(df_turns)
    train_dev_idx = sorted(df_turns.index.difference(test_idx).tolist())

    df_train = df_turns.loc[train_dev_idx].reset_index(drop=True)
    df_test = df_turns.loc[test_idx].reset_index(drop=True)

    write_json(_df_to_records(df_train), train_json)
    write_json(_df_to_records(df_test), test_json)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns-csv", default=DEFAULT_TURNS)
    parser.add_argument("--train-json", default=DEFAULT_TRAIN)
    parser.add_argument("--test-json", default=DEFAULT_TEST)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    split_train_test(
        turns_csv=args.turns_csv,
        train_json=args.train_json,
        test_json=args.test_json,
    )
