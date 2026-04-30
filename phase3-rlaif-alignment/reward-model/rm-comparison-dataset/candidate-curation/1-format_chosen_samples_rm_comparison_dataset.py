"""Initialise per-model JSON files with an empty ``predict_sft_chosen`` field.

Reads the rm_comparison_dataset_results.json produced by LLaMA-Factory for each
model, writes a copy under ``data/<model>/rm_comparison_dataset_<short>.json`` with
an empty ``predict_sft_chosen`` placeholder ready to be filled in later.

NOTE: marked "DON'T USE" in the original notebook. Kept for historical reference.
"""

from __future__ import annotations

import os

from _lib import MODELS, MODEL_TO_NAME, read_json, write_json


def _llama_factory_results_path(model: str) -> str:
    return f"../llama-factory-predict/saves/{model}/emotional_balanced/rm_comparison_dataset_results.json"


def _model_data_path(model: str) -> str:
    return f"data/{model}/rm_comparison_dataset_{MODEL_TO_NAME[model]}.json"


def initialise_chosen_predictions() -> None:
    for model in MODELS:
        comparative_data = read_json(_llama_factory_results_path(model))
        for entry in comparative_data:
            entry["predict_sft_chosen"] = [""]
        out = _model_data_path(model)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        write_json(comparative_data, out)


if __name__ == "__main__":
    initialise_chosen_predictions()
