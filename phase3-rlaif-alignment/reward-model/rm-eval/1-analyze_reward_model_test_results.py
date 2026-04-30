"""Aggregate reward-model and human-annotation predictions into two dataframes.

For each ``(model, file)`` combination this script reads
``../rm-llama-factory-training/saves/<model>/lora/predict/<file>/all_results.json`` and
loads it into a dataframe whose columns are the union of all keys.

Two dataframes are produced:

* ``df_data_rm`` – reward-model predictions (``rm_1``, ``rm_2``, ``rm_3``)
* ``df_data_ha`` – human-annotation predictions (``ha_1``, ``ha_2``, ``ha_3``)
"""

from __future__ import annotations

import json
from typing import Iterable

import pandas as pd

MODELS = [
    "gemma-2-9b-it",
    "glm-4-9b-chat-1m",
    "Meta-Llama-3-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Phi-3-small-8k-instruct",
]
RM_FILES = ["rm_1", "rm_2", "rm_3"]
HA_FILES = ["ha_1", "ha_2", "ha_3"]

LLAMA_FACTORY_ROOT = "../rm-llama-factory-training/saves"


def _load_results(model: str, file: str) -> dict:
    with open(f"{LLAMA_FACTORY_ROOT}/{model}/lora/predict/{file}/all_results.json") as f:
        return json.load(f)


def load_predictions(files: Iterable[str], models: Iterable[str] = MODELS) -> pd.DataFrame:
    rows = [_load_results(model, file) for model in models for file in files]
    columns = list(rows[0].keys())
    return pd.DataFrame(rows, columns=columns)


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_predictions(RM_FILES), load_predictions(HA_FILES)


if __name__ == "__main__":
    df_data_rm, df_data_ha = load_all()
    print("Reward model predictions:")
    print(df_data_rm)
    print("\nHuman annotation predictions:")
    print(df_data_ha)
