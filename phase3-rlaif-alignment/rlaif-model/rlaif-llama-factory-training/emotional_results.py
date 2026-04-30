"""Aggregate per-model RL prediction files (v2: SFT_D + DPO/PPO_{2,4,8}/DPR_{2,4,8}).

Two prediction groups per model:

* ``sft_d_*`` predictions over ``demonstration_data_test.json`` →
  ``demonstration_data_test_results.json``
* ``dpo_* + ppo_*_{2,4,8} + sft_dpr_*_{2,4,8}`` predictions over
  ``demonstration_prompt_rlaif_data_test.json`` →
  ``demonstration_prompt_rlaif_data_test_results.json``

Reports per-run chatbot emotion-tag accuracy (user / neutral lines exist in the
original code but are commented out).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass

MODELS = [
    "gemma-2-9b-it",
    "glm-4-9b-chat-1m",
    "Meta-Llama-3-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Phi-3-small-8k-instruct",
]
EMO_RE = re.compile(r"\(.*?\)")


@dataclass
class PredictionGroup:
    name: str
    run_names: list[str]
    source_file: str
    output_file: str


def _build_dpr_v2_run_names() -> list[str]:
    runs = ["dpo_1", "dpo_2", "dpo_3"]
    for stage in ("ppo", "sft_dpr"):
        for n in (2, 4, 8):
            for i in (1, 2, 3):
                runs.append(f"{stage}_{i}_{n}")
    return runs


GROUP_SFT_D = PredictionGroup(
    name="SFT D",
    run_names=["sft_d_1", "sft_d_2", "sft_d_3"],
    source_file="./data/demonstration_data_test.json",
    output_file="demonstration_data_test_results.json",
)
GROUP_DPR_V2 = PredictionGroup(
    name="DPR V2",
    run_names=_build_dpr_v2_run_names(),
    source_file="./data/demonstration_prompt_rlaif_data_test.json",
    output_file="demonstration_prompt_rlaif_data_test_results.json",
)
ALL_GROUPS = [GROUP_SFT_D, GROUP_DPR_V2]

REPORT_USER = False
REPORT_CHATBOT = True
REPORT_NEUTRAL = False


def _read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _safe_three_emotions(text: str) -> tuple[str, str, str]:
    found = EMO_RE.findall(text)
    found += [""] * (3 - len(found))
    return tuple(found[:3])


def _load_predictions_for_runs(model: str, run_names: list[str]) -> dict[str, list[dict]]:
    return {
        name: _read_jsonl(f"./saves/{model}/lora/predict/{name}/generated_predictions.jsonl")
        for name in run_names
    }


def _merge_predictions(source_data: list[dict], predictions: dict[str, list[dict]], model: str) -> list[dict]:
    for idx, entry in enumerate(source_data):
        entry.pop("input", None)
        entry["prompt"] = entry.pop("instruction")
        entry["instruction"] = entry.pop("system")
        entry["target"] = entry.pop("output")
        for run_name, run_results in predictions.items():
            entry[f"predict_{run_name}"] = run_results[idx]["predict"].replace("\n", "").strip()
        entry["model"] = model
    return source_data


def _write_merged(model: str, output_filename: str, merged: list[dict]) -> None:
    out_dir = f"./saves/{model}/emotional_balanced"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{output_filename}", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


def _report_agreement(merged: list[dict], run_names: list[str], only_values: bool = False) -> None:
    targets = [_safe_three_emotions(entry["target"]) for entry in merged]
    n_total = len(merged)

    for run_name in run_names:
        u_hits, c_hits, neu_hits = 0, 0, 0
        for entry, (t_user, t_chat, _) in zip(merged, targets):
            p_user, p_chat, p_neutral = _safe_three_emotions(entry[f"predict_{run_name}"])
            u_hits += p_user == t_user
            c_hits += p_chat == t_chat
            neu_hits += p_neutral == "(NEUTRAL)"

        if only_values:
            if REPORT_USER:
                print(f"\n{u_hits / n_total * 100:0.2f}%")
            if REPORT_CHATBOT:
                print(f"{c_hits / n_total * 100:0.2f}%")
            if REPORT_NEUTRAL:
                print(f"{neu_hits / n_total * 100:0.2f}%")
        else:
            print(f"\nPREDICTION MODEL {run_name}")
            if REPORT_USER:
                print(f"USER EMOTION: {u_hits / n_total * 100:0.2f}%")
            if REPORT_CHATBOT:
                print(f"CHATBOT EMOTION: {c_hits / n_total * 100:0.2f}%")
            if REPORT_NEUTRAL:
                print(f"NEUTRAL EMOTION: {neu_hits / n_total * 100:0.2f}%")


def _process_group(model: str, group: PredictionGroup, only_values: bool) -> None:
    with open(group.source_file, "r", encoding="utf-8") as f:
        source_data = json.loads(f.read())
    predictions = _load_predictions_for_runs(model, group.run_names)
    merged = _merge_predictions(source_data, predictions, model)
    _write_merged(model, group.output_file, merged)
    _report_agreement(merged, group.run_names, only_values=only_values)


def run(models: list[str] = MODELS, groups: list[PredictionGroup] = ALL_GROUPS, only_values: bool = False) -> None:
    for model in models:
        print(f"\n---------------------------------\nMODEL: {model}\n---------------------------------")
        for group in groups:
            _process_group(model, group, only_values=only_values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--only-values", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(models=args.models, only_values=args.only_values)
