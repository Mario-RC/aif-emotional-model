"""Format pairwise comparison data into the LLaMA-Factory preference JSON schema."""

from __future__ import annotations

import argparse

import pandas as pd

from _lib import with_suffix, write_json, read_json

SET_ORDER = ("sft-demonstration", "rm-prompt", "ppo-unlabeled-prompts")


def _sort_by_set(records: list[dict]) -> list[dict]:
    order = {set_name: idx for idx, set_name in enumerate(SET_ORDER)}
    return [
        record
        for _, record in sorted(
            enumerate(records),
            key=lambda item: (order.get(item[1].get("set"), len(order)), item[0]),
        )
    ]


def _value(df_uid: pd.DataFrame, key: str, default: str = "") -> str:
    if key not in df_uid:
        return default
    value = df_uid[key].values[0]
    if pd.isna(value):
        return default
    return value


def _build_entry(
    preference_id: str,
    system: str,
    history: list,
    prompt: str,
    winner: str,
    loser: str,
    dialogue_id: str,
    source_set: str | None = None,
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
    entry = {
        "conversations": conversations,
        "chosen": {"from": "gpt", "value": winner},
        "rejected": {"from": "gpt", "value": loser},
        "preference_id": preference_id,
        "dialogue_id": dialogue_id,
    }
    if source_set:
        entry["set"] = source_set
    return entry


def format_dpo_preference_dataset(is_test: bool = False, with_demonstration_concat: bool = False) -> None:
    src = f"data/{with_suffix('dpo_preference_dataset_response', 'json', is_test)}"
    out = f"data/{with_suffix('dpo_preference_dataset_1k', 'json', is_test)}"

    df = pd.read_json(src)

    entries: list[dict] = []
    preference_id_col = "PREFERENCE_ID" if "PREFERENCE_ID" in df.columns else "UID"
    for preference_id in df[preference_id_col].unique():
        df_uid = df[df[preference_id_col] == preference_id].reset_index(drop=True)
        entries.append(
            _build_entry(
                preference_id=preference_id,
                system=df_uid["SYSTEM"].values[0],
                history=df_uid["HISTORY"].values[0],
                prompt=df_uid["PROMPT"].values[0],
                winner=df_uid["WINNER_RESPONSE"].values[0],
                loser=df_uid["LOSER_RESPONSE"].values[0],
                dialogue_id=_value(df_uid, "dialogue_id", _value(df_uid, "DIALOGUE_ID")),
                source_set=_value(df_uid, "set", _value(df_uid, "SET", None)),
            )
        )

    write_json(_sort_by_set(entries), out)

    if with_demonstration_concat or is_test:
        comparison = read_json(f"data/{with_suffix('rm_preference_dataset', 'json', is_test)}")
        rlaif = read_json(out)
        write_json(
            _sort_by_set(comparison + rlaif),
            f"data/{with_suffix('dpo_preference_dataset', 'json', is_test)}",
        )



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--demo-concat", action="store_true",
                        help="Also build dpo_preference_dataset.json (train split only).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    format_dpo_preference_dataset(is_test=args.test, with_demonstration_concat=args.demo_concat)
