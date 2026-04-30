"""Aggregate RL prediction metrics across many run families into per-family DataFrames.

For each ``(family, file)`` combination, read
``../rlaif-llama-factory-training/saves/<model>/lora/predict/<file>/all_results.json``
and collect them into a single DataFrame keyed by ``(model_name, train)``.

Run families covered:

* DPO: ``dpo_1`` / ``dpo_2`` / ``dpo_3``
* PPO: ``ppo_<i>_<n>`` for i in 1..3, n in 2/4/8
* SFT-DPR: ``sft_dpr_<i>_<n>`` for i in 1..3, n in 2/4/8
* SFT-D: ``sft_d_1`` / ``sft_d_2`` / ``sft_d_3``
* RM: ``rm_1`` / ``rm_2`` / ``rm_3``
* HA: ``ha_1`` / ``ha_2`` / ``ha_3``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

MODELS = [
    "gemma-2-9b-it",
    "glm-4-9b-chat-1m",
    "Meta-Llama-3-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Phi-3-small-8k-instruct",
]
MODEL_NAMES = ["gemma2", "glm4", "llama3", "mistral", "phi3"]
LLAMA_FACTORY_ROOT = "../rlaif-llama-factory-training/saves"


@dataclass(frozen=True)
class RunFamily:
    name: str
    files: list[str]
    train_labels: list[str]


def _ppo_family(scale: int) -> RunFamily:
    files = [f"ppo_{i}_{scale}" for i in (1, 2, 3)]
    return RunFamily(name=f"ppo_{scale}", files=files, train_labels=files)


def _sft_dpr_family(scale: int) -> RunFamily:
    files = [f"sft_dpr_{i}_{scale}" for i in (1, 2, 3)]
    return RunFamily(name=f"sft_dpr_{scale}", files=files, train_labels=["sft_dpr_1", "sft_dpr_2", "sft_dpr_3"])


FAMILIES: list[RunFamily] = [
    RunFamily(name="dpo", files=["dpo_1", "dpo_2", "dpo_3"], train_labels=["dpo_1", "dpo_2", "dpo_3"]),
    _ppo_family(2), _ppo_family(4), _ppo_family(8),
    _sft_dpr_family(2), _sft_dpr_family(4), _sft_dpr_family(8),
    RunFamily(name="sft_d", files=["sft_d_1", "sft_d_2", "sft_d_3"], train_labels=["sft_d_1", "sft_d_2", "sft_d_3"]),
    RunFamily(name="rm", files=["rm_1", "rm_2", "rm_3"], train_labels=["rm_1", "rm_2", "rm_3"]),
    RunFamily(name="ha", files=["ha_1", "ha_2", "ha_3"], train_labels=["ha_1", "ha_2", "ha_3"]),
]


def _load_all_results(model: str, file: str) -> dict:
    with open(f"{LLAMA_FACTORY_ROOT}/{model}/lora/predict/{file}/all_results.json") as f:
        return json.load(f)


def load_family_dataframe(family: RunFamily, models: Iterable[str] = MODELS) -> pd.DataFrame:
    """Load every (file, model) pair for ``family`` into a single labelled DataFrame."""
    rows: list[dict] = []
    train_per_row: list[str] = []
    model_per_row: list[str] = []

    for file_idx, file in enumerate(family.files):
        for model_idx, model in enumerate(models):
            rows.append(_load_all_results(model, file))
            train_per_row.append(family.train_labels[file_idx])
            model_per_row.append(MODEL_NAMES[model_idx])

    df = pd.DataFrame(rows, columns=list(rows[0].keys()))
    df.insert(0, "model_name", model_per_row)
    df.insert(1, "train", train_per_row)
    return df


def load_all_families(models: Iterable[str] = MODELS) -> dict[str, pd.DataFrame]:
    return {family.name: load_family_dataframe(family, models) for family in FAMILIES}


if __name__ == "__main__":
    dfs = load_all_families()
    for name, df in dfs.items():
        print(f"\n--- {name} ---")
        print(df)
