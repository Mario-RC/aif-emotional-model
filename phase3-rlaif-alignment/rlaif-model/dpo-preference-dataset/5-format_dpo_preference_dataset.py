"""Format pairwise comparison data into the LLaMA-Factory preference JSON schema."""

from __future__ import annotations

import argparse

import pandas as pd

from _lib import with_suffix, write_json, read_json


def _value(df_uid: pd.DataFrame, key: str, default: str = "") -> str:
    if key not in df_uid:
        return default
    value = df_uid[key].values[0]
    if pd.isna(value):
        return default
    return value


def _build_entry(
    uid: str,
    system: str,
    history: list,
    prompt: str,
    winner: str,
    loser: str,
    dialogue_id: str,
) -> dict:
    conversations = [
        {"from": "system", "value": system},
        {"from": "human", "value": history[0][0]},
        {"from": "gpt", "value": history[0][1]},
        {"from": "human", "value": history[1][0]},
        {"from": "gpt", "value": history[1][1]},
        {"from": "human", "value": history[2][0]},
        {"from": "gpt", "value": history[2][1]},
        {"from": "human", "value": prompt},
    ]
    return {
        "conversations": conversations,
        "chosen": {"from": "gpt", "value": winner},
        "rejected": {"from": "gpt", "value": loser},
        "uid": uid,
        "dialogue_id": dialogue_id,
    }


def format_dpo_preference_dataset(is_test: bool = False, with_demonstration_concat: bool = False) -> None:
    src = f"data/{with_suffix('dpo_preference_dataset_response', 'json', is_test)}"
    out = f"data/{with_suffix('dpo_preference_dataset_1k', 'json', is_test)}"

    df = pd.read_json(src)

    entries: list[dict] = []
    for uid in df["UID"].unique():
        df_uid = df[df["UID"] == uid].reset_index(drop=True)
        entries.append(
            _build_entry(
                uid=uid,
                system=df_uid["SYSTEM"].values[0],
                history=df_uid["HISTORY"].values[0],
                prompt=df_uid["PROMPT"].values[0],
                winner=df_uid["WINNER_RESPONSE"].values[0],
                loser=df_uid["LOSER_RESPONSE"].values[0],
                dialogue_id=_value(df_uid, "dialogue_id", _value(df_uid, "DIALOGUE_ID")),
            )
        )

    write_json(entries, out)

    if with_demonstration_concat and not is_test:
        comparison = read_json("data/rm_preference_dataset.json")
        rlaif = read_json(out)
        write_json(comparison + rlaif, "data/dpo_preference_dataset.json")



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--demo-concat", action="store_true",
                        help="Also build dpo_preference_dataset.json (train split only).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    format_dpo_preference_dataset(is_test=args.test, with_demonstration_concat=args.demo_concat)
