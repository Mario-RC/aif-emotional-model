"""Format the pairwise comparison data into the LLaMA-Factory reward-model JSON schema.

Output schema (one entry per UID)::

    {
      "conversations": [
        {"from": "system", "value": <system>},
        {"from": "human",  "value": <history[0][0]>},
        {"from": "gpt",    "value": <history[0][1]>},
        ...
        {"from": "human",  "value": <prompt>}
      ],
      "chosen":   {"from": "gpt", "value": <winner_response>},
      "rejected": {"from": "gpt", "value": <loser_response>},
      "uid": <uid>
    }
"""

from __future__ import annotations

import argparse

import pandas as pd

from _lib import with_suffix, write_json


def _build_entry(uid: str, system: str, history: list, prompt: str, winner: str, loser: str) -> dict:
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
    }


def format_rm_preference_dataset(is_test: bool = False) -> None:
    src = f"data/{with_suffix('rm_preference_dataset_response', 'json', is_test)}"
    out = f"data/{with_suffix('rm_preference_dataset', 'json', is_test)}"

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
            )
        )

    write_json(entries, out)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    format_rm_preference_dataset(is_test=args.test)
