"""Preprocess raw candidate-curation output into the canonical RM Preference Dataset format.

Steps:
1. Read the upstream ``rm_comparison_dataset[*test].json``.
2. Rename keys to ``history`` / ``prompt`` / ``target`` / ``predict_1..9`` / etc.
3. Save the canonical original responses as ``rm_preference_dataset_original[*test].json``.
4. Reformat each response by replacing literal emotion tags with ``(EMPATHY)`` /
   ``(QUESTION)`` placeholders via :func:`empathy_question_modify`.

Run ``python 1-preprocess_rm_preference_dataset.py`` for the train split or
``python 1-preprocess_rm_preference_dataset.py --test`` for the test split.
"""

from __future__ import annotations

import argparse
from copy import deepcopy

from _lib import empathy_question_modify, read_json, with_suffix, write_json

UPSTREAM_DIR = "../rm-comparison-dataset/candidate-curation/data"


def _rename_keys(rm_preference_dataset: list[dict]) -> list[dict]:
    """Rename the old key names to the canonical ``history``/``predict_*`` schema."""
    for entry in rm_preference_dataset:
        keys = list(entry.keys())

        def clean(text: str) -> str:
            return text.replace("\n", "").replace("  ", " ").strip()

        history = entry.get(keys[1])
        for turn in history:
            turn[0] = clean(turn[0])
            turn[1] = clean(turn[1])
        entry["history"] = history
        entry["prompt"] = clean(entry.get(keys[2]))
        entry["target"] = clean(entry.get(keys[3]))
        for i in range(1, 10):
            entry[f"predict_{i}"] = clean(entry.get(keys[3 + i]))

        # Drop the original predict_sft_* columns now that they have been copied.
        keys_after = list(entry.keys())
        for old in keys_after[4:13]:
            entry.pop(old, None)

        if not entry.get("dialogue_id"):
            raise KeyError("Preference row is missing dialogue_id.")

        for tail in ("predict_sft_modified_label", "scores", "dialogue_id"):
            if tail in entry:
                entry[tail] = entry.pop(tail)
    return rm_preference_dataset


def _reformat_responses(rm_preference_dataset: list[dict]) -> list[dict]:
    """Replace literal emotion tags with role tags (``(EMPATHY)`` / ``(QUESTION)``)."""
    for entry in rm_preference_dataset:
        for turn in entry["history"]:
            turn[1] = empathy_question_modify(turn[1])
        entry["target"] = empathy_question_modify(entry["target"])
        for i in range(1, 10):
            key = f"predict_{i}"
            entry[key] = empathy_question_modify(entry[key])
    return rm_preference_dataset


def preprocess(is_test: bool = False) -> None:
    upstream_file = f"{UPSTREAM_DIR}/{with_suffix('rm_comparison_dataset', 'json', is_test)}"
    out_file = f"data/{with_suffix('rm_preference_dataset', 'json', is_test)}"
    original_file = f"data/{with_suffix('rm_preference_dataset_original', 'json', is_test)}"

    data = _rename_keys(read_json(upstream_file))
    write_json(data, original_file)

    rating_input = _reformat_responses(deepcopy(data))
    write_json(rating_input, out_file)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess(is_test=args.test)
