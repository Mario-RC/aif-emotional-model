"""Aggregate per-model SFT vs foundation prediction files and report emotion-tag agreement.

For each model in :data:`MODELS`, this script:
1. Loads the foundation-model and SFT predictions
   (``./saves/<model>/foundation/predict/generated_predictions.jsonl`` and
   ``./saves/<model>/predict/sft/generated_predictions.jsonl``)
   plus the source ``./data/sft_demonstration_dataset_test.json``.
2. Merges the predictions into a single record per dialogue and saves it under
   ``./saves/<model>/emotional_balanced/demonstration_data_emotional_balanced_test_results.json``.
3. Prints per-model accuracy of the user / chatbot / neutral emotion tags
   plus the mean number of emotion tags per turn (foundation only).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from statistics import mean
from typing import Iterable

MODELS = [
    "chatglm3-6b-32k",
    "glm-4-9b-chat-1m",
    "gemma-1.1-7b-it",
    "gemma-2-9b-it",
    "internlm2-chat-7b",
    "internlm2_5-7b-chat",
    "Llama-2-7b-chat-hf",
    "Meta-Llama-3-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Phi-3-mini-4k-instruct",
    "Phi-3-small-8k-instruct",
]

FOUNDATION_PATH_TPL = "./saves/{model}/foundation/predict/generated_predictions.jsonl"
SFT_PATH_TPL = "./saves/{model}/predict/sft/generated_predictions.jsonl"
TEST_DATA_PATH = "./data/sft_demonstration_dataset_test.json"

EMO_RE = re.compile(r"\(.*?\)")


def _read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _safe_three_emotions(text: str) -> tuple[str, str, str]:
    found = EMO_RE.findall(text)
    found += [""] * (3 - len(found))
    return tuple(found[:3])


def load_predictions(model: str) -> tuple[list[dict], list[dict]]:
    foundation = _read_jsonl(FOUNDATION_PATH_TPL.format(model=model))
    sft = _read_jsonl(SFT_PATH_TPL.format(model=model))
    return foundation, sft


def merge_predictions(
    test_data: list[dict], sft: list[dict], foundation: list[dict]
) -> list[dict]:
    """Mutate ``test_data`` in place adding ``predict_sft`` and ``predict_foundation``."""
    for idx, (entry, sft_pred, fnd_pred) in enumerate(zip(test_data, sft, foundation)):
        entry.pop("input", None)
        entry["input"] = entry.pop("instruction")
        entry["instruction"] = entry.pop("system")
        entry["target"] = entry.pop("output")
        entry["predict_sft"] = sft_pred["predict"].replace("\n", "").strip()
        entry["predict_foundation"] = fnd_pred["predict"].replace("\n", "").strip()
    return test_data


def write_merged(model: str, merged: list[dict]) -> None:
    out_dir = f"./saves/{model}/emotional_balanced"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/demonstration_data_emotional_balanced_test_results.json",
              "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


def report_agreement(merged: list[dict], only_values: bool = False) -> None:
    sft_user, sft_chat, sft_neutral = 0, 0, 0
    fnd_user, fnd_chat, fnd_neutral = 0, 0, 0
    fnd_emotion_counts: list[int] = []

    for entry in merged:
        t_user, t_chat, _ = _safe_three_emotions(entry["target"])
        s_user, s_chat, s_neutral = _safe_three_emotions(entry["predict_sft"])
        f_user, f_chat, f_neutral = _safe_three_emotions(entry["predict_foundation"])
        fnd_emotion_counts.append(len(EMO_RE.findall(entry["predict_foundation"])))

        sft_user += s_user == t_user
        sft_chat += s_chat == t_chat
        sft_neutral += s_neutral == "(NEUTRAL)"
        fnd_user += f_user == t_user
        fnd_chat += f_chat == t_chat
        fnd_neutral += f_neutral == "(NEUTRAL)"

    n = len(merged)
    if only_values:
        print(f"{fnd_user / n * 100:0.2f}%")
        print(f"{fnd_chat / n * 100:0.2f}%")
        print(f"{fnd_neutral / n * 100:0.2f}%")
        print(f"{mean(fnd_emotion_counts):0.2f}")
        print(f"\n\n{sft_user / n * 100:0.2f}%")
        print(f"{sft_chat / n * 100:0.2f}%")
        print(f"{sft_neutral / n * 100:0.2f}%")
    else:
        print("\nPREDICTION FOUNDATION MODEL")
        print(f"USER EMOTION: {fnd_user / n * 100:0.2f}%")
        print(f"CHATBOT EMOTION: {fnd_chat / n * 100:0.2f}%")
        print(f"NEUTRAL EMOTION: {fnd_neutral / n * 100:0.2f}%")
        print(f"MEAN EMOTIONS PER TURN: {mean(fnd_emotion_counts):0.2f}")
        print("\nPREDICTION SFT MODEL")
        print(f"USER EMOTION: {sft_user / n * 100:0.2f}%")
        print(f"CHATBOT EMOTION: {sft_chat / n * 100:0.2f}%")
        print(f"NEUTRAL EMOTION: {sft_neutral / n * 100:0.2f}%")


def run(models: Iterable[str] = MODELS, only_values: bool = True) -> None:
    for model in models:
        with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
            test_data = json.loads(f.read())

        foundation, sft = load_predictions(model)
        merged = merge_predictions(test_data, sft, foundation)
        write_merged(model, merged)

        print(f"\n---------------------------------\nMODEL: {model}\n---------------------------------")
        report_agreement(merged, only_values=only_values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--verbose", action="store_true", help="Print labelled metrics instead of bare values.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(models=args.models, only_values=not args.verbose)
