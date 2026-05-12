"""Split the per-turn prompt data into emotion-balanced train/test JSON files.

Test split: 2 samples per (EMOTION_RESPONSE_1, EMOTION_RESPONSE_2) pair, taken
from the last turn of each dialogue (UID ending in "6"); the remaining dialogues
form the train_dev split. Both splits are exported as LLaMA-Factory-style JSON.

Outputs:
    data/rm_prompt_dataset_emotional_balanced.json
    data/rm_prompt_dataset_emotional_balanced_test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _lib import build_dialogue_record, write_json

DEFAULT_TURNS = "data/rm_prompt_dataset_turns.csv"
DEFAULT_TRAIN = "data/rm_prompt_dataset_emotional_balanced.json"
DEFAULT_TEST = "data/rm_prompt_dataset_emotional_balanced_test.json"
DEFAULT_TEST_DIDS = "data/rm_prompt_dataset_test_dids.json"
N_TIMES_PER_PAIR = 2


def _select_test_indices_from_dids(df_turns: pd.DataFrame, test_dids_path: str) -> list[int]:
    """Load the canonical annotated test split and expand it to full dialogues."""
    with open(test_dids_path, encoding="utf-8") as f:
        test_dids = json.load(f)

    missing = sorted(set(test_dids).difference(df_turns["DID"].unique()))
    if missing:
        raise ValueError(f"Test DID list contains missing DID values: {missing[:5]}")

    test_set = df_turns.index[df_turns["DID"].isin(test_dids)].tolist()
    expected_rows = len(test_dids) * 4
    if len(test_set) != expected_rows:
        raise ValueError(f"Expected {expected_rows} test rows for {len(test_dids)} DID values, found {len(test_set)}.")

    return sorted(test_set)


def _select_test_indices_sampled(df_turns: pd.DataFrame, random_seed: int | None = None) -> list[int]:
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
            pair_indices.append(pair_df.sample(n=N_TIMES_PER_PAIR, random_state=random_seed).index)

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
    test_dids_json: str | None = DEFAULT_TEST_DIDS,
    random_seed: int | None = None,
) -> None:
    df_turns = pd.read_csv(turns_csv, encoding="utf-8")

    if test_dids_json and Path(test_dids_json).exists():
        test_idx = _select_test_indices_from_dids(df_turns, test_dids_json)
    else:
        test_idx = _select_test_indices_sampled(df_turns, random_seed=random_seed)
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
    parser.add_argument("--test-dids-json", default=DEFAULT_TEST_DIDS)
    parser.add_argument("--random-seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    split_train_test(
        turns_csv=args.turns_csv,
        train_json=args.train_json,
        test_json=args.test_json,
        test_dids_json=args.test_dids_json,
        random_seed=args.random_seed,
    )
