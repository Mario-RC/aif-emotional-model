"""Aggregate per-model RLAIF prediction files.

By default the script reuses existing
``ppo_unlabeled_prompts_dataset_test_results.json`` files when present. Use
``--regenerate-from-predictions`` to rebuild them from the current
``saves/<model>/predict/*`` outputs.

Reports per-run chatbot emotion-tag accuracy (user / neutral lines exist in the
original code but are commented out).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

MODELS = [
    "gemma-2-9b-it",
    "glm-4-9b-chat-1m",
    "Meta-Llama-3-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Phi-3-small-8k-instruct",
]
EMO_RE = re.compile(r"\(.*?\)")
LEGACY_PREDICTION_ALIASES = {
    "sft_dpr_1_4": "sft_dpr_1ep",
    "ppo_1_4": "ppo_1ep",
    "dpo_3": "dpo_3ep",
}


@dataclass
class PredictionGroup:
    name: str
    run_names: list[str]
    source_file: str
    output_file: str


def _build_rlaif_run_names() -> list[str]:
    runs = []
    for stage in ("dpo", "ppo", "sft_dpr"):
        for epoch in (1, 2, 3):
            runs.append(f"{stage}_{epoch}ep")
    return runs


GROUP_PPO_UNLABELED_PROMPTS = PredictionGroup(
    name="PPO unlabeled prompts",
    run_names=_build_rlaif_run_names(),
    source_file="./data/ppo_unlabeled_prompts_dataset_test.json",
    output_file="ppo_unlabeled_prompts_dataset_test_results.json",
)
ALL_GROUPS = [GROUP_PPO_UNLABELED_PROMPTS]

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
) -> dict:
    return {
        "model": model,
        "dataset": dataset,
        "prediction_model": prediction_model,
        "total_examples": total,
        "results_file": results_file,
        "user_emotion": {"correct": user_hits, "percentage": _pct(user_hits, total)},
        "chatbot_emotion": {"correct": chatbot_hits, "percentage": _pct(chatbot_hits, total)},
        "neutral_emotion": {"correct": neutral_hits, "percentage": _pct(neutral_hits, total)},
    }


def _load_predictions_for_runs(model: str, run_names: list[str]) -> dict[str, list[dict]]:
    return {
        name: _read_jsonl(f"./saves/{model}/predict/{name}/generated_predictions.jsonl")
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
        for legacy_name, current_name in LEGACY_PREDICTION_ALIASES.items():
            current_key = f"predict_{current_name}"
            if current_key in entry:
                entry[f"predict_{legacy_name}"] = entry[current_key]
        entry["model"] = model
    return source_data


def _write_merged(model: str, output_filename: str, merged: list[dict]) -> str:
    out_dir = f"./saves/{model}/emotional_balanced"
    os.makedirs(out_dir, exist_ok=True)
    output_path = f"{out_dir}/{output_filename}"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return output_path


def _summary_path(model: str) -> str:
    return f"./saves/{model}/emotional_balanced/emotional_results_summary.json"


def _write_summary(model: str, metrics: list[dict]) -> str:
    output_path = _summary_path(model)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"model": model, "metrics": metrics}, f, ensure_ascii=False, indent=2)
    return output_path


def _prediction_run_names_in_data(merged: list[dict]) -> list[str]:
    if not merged:
        return []
    return sorted(key.removeprefix("predict_") for key in merged[0] if key.startswith("predict_"))


def _source_matches_output(source_data: list[dict], merged_data: list[dict]) -> bool:
    transformed = []
    for entry in source_data:
        item = dict(entry)
        item.pop("input", None)
        item["prompt"] = item.pop("instruction")
        item["instruction"] = item.pop("system")
        item["target"] = item.pop("output")
        transformed.append(item)

    base_keys = ("instruction", "history", "prompt", "target", "dialogue_id")
    return len(transformed) == len(merged_data) and all(
        all(new_entry.get(key) == old_entry.get(key) for key in base_keys)
        for new_entry, old_entry in zip(transformed, merged_data)
    )


def _load_existing_output(model: str, group: PredictionGroup, source_data: list[dict]) -> list[dict] | None:
    output_path = Path(f"./saves/{model}/emotional_balanced/{group.output_file}")
    if not output_path.exists():
        return None

    with output_path.open("r", encoding="utf-8") as f:
        merged_data = json.load(f)

    if not _source_matches_output(source_data, merged_data):
        raise ValueError(
            f"{output_path} does not match {group.source_file}; refusing to reuse changed data."
        )

    return merged_data


def _report_agreement(
    model: str,
    group: PredictionGroup,
    merged: list[dict],
    run_names: list[str],
    only_values: bool = False,
) -> list[dict]:
    targets = [_safe_three_emotions(entry["target"]) for entry in merged]
    n_total = len(merged)
    dataset = Path(group.source_file).stem
    results_file = f"./saves/{model}/emotional_balanced/{group.output_file}"
    metrics: list[dict] = []

    for run_name in run_names:
        u_hits, c_hits, neu_hits = 0, 0, 0
        for entry, (t_user, t_chat, _) in zip(merged, targets):
            p_user, p_chat, p_neutral = _safe_three_emotions(entry[f"predict_{run_name}"])
            u_hits += p_user == t_user
            c_hits += p_chat == t_chat
            neu_hits += p_neutral == "(NEUTRAL)"

        if only_values:
            if REPORT_USER:
                print(f"\n{_pct(u_hits, n_total):0.2f}%")
            if REPORT_CHATBOT:
                print(f"{_pct(c_hits, n_total):0.2f}%")
            if REPORT_NEUTRAL:
                print(f"{_pct(neu_hits, n_total):0.2f}%")
        else:
            print(f"\nPREDICTION MODEL {run_name}")
            if REPORT_USER:
                print(f"USER EMOTION: {_pct(u_hits, n_total):0.2f}%")
            if REPORT_CHATBOT:
                print(f"CHATBOT EMOTION: {_pct(c_hits, n_total):0.2f}%")
            if REPORT_NEUTRAL:
                print(f"NEUTRAL EMOTION: {_pct(neu_hits, n_total):0.2f}%")

        metrics.append(
            _metric_row(model, dataset, run_name, n_total, u_hits, c_hits, neu_hits, results_file)
        )

    return metrics


def _process_group(model: str, group: PredictionGroup, only_values: bool, regenerate_from_predictions: bool) -> list[dict]:
    with open(group.source_file, "r", encoding="utf-8") as f:
        source_data = json.loads(f.read())

    merged = None if regenerate_from_predictions else _load_existing_output(model, group, source_data)
    if merged is None:
        predictions = _load_predictions_for_runs(model, group.run_names)
        merged = _merge_predictions(source_data, predictions, model)
        _write_merged(model, group.output_file, merged)
        report_run_names = group.run_names
    else:
        report_run_names = _prediction_run_names_in_data(merged)

    return _report_agreement(model, group, merged, report_run_names, only_values=only_values)


def run(
    models: list[str] = MODELS,
    groups: list[PredictionGroup] = ALL_GROUPS,
    only_values: bool = False,
    regenerate_from_predictions: bool = False,
) -> None:
    for model in models:
        print(f"\n---------------------------------\nMODEL: {model}\n---------------------------------")
        model_metrics: list[dict] = []
        for group in groups:
            model_metrics.extend(
                _process_group(
                    model,
                    group,
                    only_values=only_values,
                    regenerate_from_predictions=regenerate_from_predictions,
                )
            )
        summary_file = _write_summary(model, model_metrics)
        if not only_values:
            print(f"\nSaved emotional summary: {summary_file}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--only-values", action="store_true")
    parser.add_argument(
        "--regenerate-from-predictions",
        action="store_true",
        help="Rebuild from current saves/<model>/predict/* files instead of preserving migrated historical results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        models=args.models,
        only_values=args.only_values,
        regenerate_from_predictions=args.regenerate_from_predictions,
    )
