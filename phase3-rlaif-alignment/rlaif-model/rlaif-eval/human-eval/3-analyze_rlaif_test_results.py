"""Aggregate RL prediction metrics across the current RL run families.

For each ``(family, file)`` combination, read
``../rlaif-llama-factory-training/saves/<model>/predict/<file>/all_results.json``
and collect them into a single DataFrame keyed by ``(model_name, train)``.

Run families covered:

* DPO: ``dpo_1ep`` / ``dpo_2ep`` / ``dpo_3ep``
* PPO: ``ppo_1ep`` / ``ppo_2ep`` / ``ppo_3ep``
* SFT-DPR: ``sft_dpr_1ep`` / ``sft_dpr_2ep`` / ``sft_dpr_3ep``
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
LLAMA_FACTORY_ROOT = "../../rlaif-llama-factory-training/saves"


@dataclass(frozen=True)
class RunFamily:
    name: str
    files: list[str]
    train_labels: list[str]


FAMILIES: list[RunFamily] = [
    RunFamily(name="dpo", files=["dpo_1ep", "dpo_2ep", "dpo_3ep"], train_labels=["dpo_1ep", "dpo_2ep", "dpo_3ep"]),
    RunFamily(name="ppo", files=["ppo_1ep", "ppo_2ep", "ppo_3ep"], train_labels=["ppo_1ep", "ppo_2ep", "ppo_3ep"]),
    RunFamily(name="sft_dpr", files=["sft_dpr_1ep", "sft_dpr_2ep", "sft_dpr_3ep"], train_labels=["sft_dpr_1ep", "sft_dpr_2ep", "sft_dpr_3ep"]),
]


def _load_all_results(model: str, file: str) -> dict:
    with open(f"{LLAMA_FACTORY_ROOT}/{model}/predict/{file}/all_results.json") as f:
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
