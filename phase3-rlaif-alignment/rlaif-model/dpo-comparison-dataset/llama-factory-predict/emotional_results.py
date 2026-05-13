"""Aggregate per-model SFT prediction files and report emotion-tag agreement.

For each (split, model) combination this script:
1. Loads ``./saves/<model>/predict/<split>/predict_<n>/generated_predictions.jsonl``
   for n in 0..7 plus the matching ``./data/ppo_unlabeled_prompts_dataset_1k[_test].json`` source.
2. Merges the predictions into a single record per dialogue and saves it under
   ``./saves/<model>/emotional_balanced/dpo_comparison_dataset[_test]_results.json``.
3. Prints the per-model accuracy of the user/chatbot/neutral emotion tags
   against the target.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Iterable

MODELS = [
    "gemma-2-9b-it",
    "glm-4-9b-chat-1m",
    "Meta-Llama-3-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Phi-3-small-8k-instruct",
]
SPLITS = ["train", "test"]
N_PREDICTIONS = 8
EMO_RE = re.compile(r"\(.*?\)")


def _read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_predictions(model: str, split: str) -> list[list[dict]]:
    return [
        _read_jsonl(f"./saves/{model}/predict/{split}/predict_{n}/generated_predictions.jsonl")
        for n in range(N_PREDICTIONS)
    ]


def load_source_data(split: str) -> list[dict]:
    suffix = "_test" if split == "test" else ""
    with open(f"./data/ppo_unlabeled_prompts_dataset_1k{suffix}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def merge_predictions(prompt_data: list[dict], predictions: list[list[dict]], model: str) -> list[dict]:
    for idx, entry in enumerate(prompt_data):
        entry.pop("input", None)
        entry["prompt"] = entry.pop("instruction")
        entry["instruction"] = entry.pop("system")
        entry["target"] = entry.pop("output")
        for n in range(N_PREDICTIONS):
            entry[f"predict_sft_{n}"] = predictions[n][idx]["predict"].replace("\n", "").strip()
        entry["model"] = model
    return prompt_data


def write_merged(model: str, split: str, merged: list[dict]) -> None:
    out_dir = f"./saves/{model}/emotional_balanced"
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_test" if split == "test" else ""
    with open(f"{out_dir}/dpo_comparison_dataset{suffix}_results.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


def _safe_three_emotions(text: str) -> tuple[str, str, str]:
    found = EMO_RE.findall(text)
    found += [""] * (3 - len(found))
    return tuple(found[:3])


def report_agreement(prompt_data: list[dict], n_predict: int = N_PREDICTIONS, only_values: bool = False) -> None:
    targets = [_safe_three_emotions(entry["target"]) for entry in prompt_data]

    for n in range(n_predict):
        u_hits, c_hits, neu_hits = 0, 0, 0
        for entry, (t_user, t_chat, _) in zip(prompt_data, targets):
            p_user, p_chat, p_neutral = _safe_three_emotions(entry[f"predict_sft_{n}"])
            u_hits += p_user == t_user
            c_hits += p_chat == t_chat
            neu_hits += p_neutral == "(NEUTRAL)"
        n_total = len(prompt_data)
        if only_values:
            print(f"\n{u_hits / n_total * 100:0.2f}%")
            print(f"{c_hits / n_total * 100:0.2f}%")
            print(f"{neu_hits / n_total * 100:0.2f}%")
        else:
            print(f"\nPREDICTION {n} SFT MODEL")
            print(f"USER EMOTION: {u_hits / n_total * 100:0.2f}%")
            print(f"CHATBOT EMOTION: {c_hits / n_total * 100:0.2f}%")
            print(f"NEUTRAL EMOTION: {neu_hits / n_total * 100:0.2f}%")


def run(splits: Iterable[str] = SPLITS, models: Iterable[str] = MODELS, only_values: bool = False) -> None:
    for split in splits:
        for model in models:
            print(f"\n---------------------------------\n{split} MODEL: {model}\n---------------------------------")
            predictions = load_predictions(model, split)
            prompt_data = load_source_data(split)
            merged = merge_predictions(prompt_data, predictions, model)
            write_merged(model, split, merged)
            report_agreement(merged, only_values=only_values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--only-values", action="store_true",
                        help="Print bare percentages without labels (matches the ``only_values=True`` notebook flag).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(splits=args.splits, models=args.models, only_values=args.only_values)
