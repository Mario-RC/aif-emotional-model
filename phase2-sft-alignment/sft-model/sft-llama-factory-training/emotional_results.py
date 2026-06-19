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


def _pct(count: int, total: int) -> float:
    return round(count / total * 100, 2) if total else 0.0


def _metric_row(
    model: str,
    dataset: str,
    prediction_model: str,
    total: int,
    user_hits: int,
    chatbot_hits: int,
    neutral_hits: int,
    results_file: str,
    mean_emotions_per_turn: float | None = None,
) -> dict:
    row = {
        "model": model,
        "dataset": dataset,
        "prediction_model": prediction_model,
        "total_examples": total,
        "results_file": results_file,
        "user_emotion": {"correct": user_hits, "percentage": _pct(user_hits, total)},
        "chatbot_emotion": {"correct": chatbot_hits, "percentage": _pct(chatbot_hits, total)},
        "neutral_emotion": {"correct": neutral_hits, "percentage": _pct(neutral_hits, total)},
    }
    if mean_emotions_per_turn is not None:
        row["mean_emotions_per_turn"] = round(mean_emotions_per_turn, 2)
    return row


def load_predictions(model: str) -> tuple[list[dict], list[dict]]:
    foundation = _read_jsonl(FOUNDATION_PATH_TPL.format(model=model))
    sft = _read_jsonl(SFT_PATH_TPL.format(model=model))
    return foundation, sft


def merge_predictions(
    test_data: list[dict], sft: list[dict], foundation: list[dict]
) -> list[dict]:
    """Return records with both SFT and foundation predictions attached."""
    merged = []
    for idx, (entry, sft_pred, fnd_pred) in enumerate(zip(test_data, sft, foundation)):
        entry.pop("input", None)
        entry["input"] = entry.pop("instruction")
        entry["instruction"] = entry.pop("system")
        entry["target"] = entry.pop("output")
        entry["predict_sft"] = sft_pred["predict"].replace("\n", "").strip()
        entry["predict_foundation"] = fnd_pred["predict"].replace("\n", "").strip()
        merged.append(entry)
    return merged


def write_merged(model: str, merged: list[dict]) -> str:
    out_dir = f"./saves/{model}/emotional_balanced"
    os.makedirs(out_dir, exist_ok=True)
    output_path = f"{out_dir}/demonstration_data_emotional_balanced_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return output_path


def write_summary(model: str, metrics: list[dict]) -> str:
    out_dir = f"./saves/{model}/emotional_balanced"
    os.makedirs(out_dir, exist_ok=True)
    output_path = f"{out_dir}/emotional_results_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"model": model, "metrics": metrics}, f, ensure_ascii=False, indent=2)
    return output_path


def report_agreement(model: str, merged: list[dict], results_file: str, only_values: bool = False) -> list[dict]:
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
    foundation_mean = mean(fnd_emotion_counts) if fnd_emotion_counts else 0.0
    if only_values:
        print(f"{_pct(fnd_user, n):0.2f}%")
        print(f"{_pct(fnd_chat, n):0.2f}%")
        print(f"{_pct(fnd_neutral, n):0.2f}%")
        print(f"{foundation_mean:0.2f}")
        print(f"\n\n{_pct(sft_user, n):0.2f}%")
        print(f"{_pct(sft_chat, n):0.2f}%")
        print(f"{_pct(sft_neutral, n):0.2f}%")
    else:
        print("\nPREDICTION FOUNDATION MODEL")
        print(f"USER EMOTION: {_pct(fnd_user, n):0.2f}%")
        print(f"CHATBOT EMOTION: {_pct(fnd_chat, n):0.2f}%")
        print(f"NEUTRAL EMOTION: {_pct(fnd_neutral, n):0.2f}%")
        print(f"MEAN EMOTIONS PER TURN: {foundation_mean:0.2f}")
        print("\nPREDICTION SFT MODEL")
        print(f"USER EMOTION: {_pct(sft_user, n):0.2f}%")
        print(f"CHATBOT EMOTION: {_pct(sft_chat, n):0.2f}%")
        print(f"NEUTRAL EMOTION: {_pct(sft_neutral, n):0.2f}%")

    dataset = "sft_demonstration_dataset_test"
    return [
        _metric_row(model, dataset, "foundation", n, fnd_user, fnd_chat, fnd_neutral, results_file, foundation_mean),
        _metric_row(model, dataset, "sft", n, sft_user, sft_chat, sft_neutral, results_file),
    ]


def run(models: Iterable[str] = MODELS, only_values: bool = True) -> None:
    for model in models:
        with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
            test_data = json.loads(f.read())

        foundation, sft = load_predictions(model)
        merged = merge_predictions(test_data, sft, foundation)
        results_file = write_merged(model, merged)

        print(f"\n---------------------------------\nMODEL: {model}\n---------------------------------")
        metrics = report_agreement(model, merged, results_file, only_values=only_values)
        summary_file = write_summary(model, metrics)
        if not only_values:
            print(f"\nSaved emotional summary: {summary_file}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--verbose", action="store_true", help="Print labelled metrics instead of bare values.")
    parser.add_argument(
        "--regenerate-from-predictions",
        action="store_true",
        help="Accepted for CLI compatibility; phase2 always rebuilds results from current prediction files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(models=args.models, only_values=not args.verbose)
