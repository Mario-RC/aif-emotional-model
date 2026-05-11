"""Aggregate reward-model and human-annotation prediction metrics.

For each ``(model, file)`` combination this script reads
``../../rm-llama-factory-training/saves/<model>/predict/<file>/all_results.json`` and
loads it into rows whose columns are the union of all keys.

Two tables are produced:

* ``data_rm`` – reward-model predictions (``rm_1``, ``rm_2``, ``rm_3``)
* ``data_ha`` – human-annotation predictions (``ha_1``, ``ha_2``, ``ha_3``)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

MODELS = [
    "gemma-2-9b-it",
    "glm-4-9b-chat-1m",
    "Meta-Llama-3-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Phi-3-small-8k-instruct",
]
RM_FILES = ["rm_1", "rm_2", "rm_3"]
HA_FILES = ["ha_1", "ha_2", "ha_3"]

REWARD_MODEL_ROOT = Path(__file__).resolve().parents[2]
LLAMA_FACTORY_SAVES = REWARD_MODEL_ROOT / "rm-llama-factory-training" / "saves"


def _load_results(model: str, file: str) -> dict:
    path = LLAMA_FACTORY_SAVES / model / "predict" / file / "all_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction results: {path}")

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_predictions(files: Iterable[str], models: Iterable[str] = MODELS) -> list[dict]:
    return [
        {"model": model, "run": file, **_load_results(model, file)}
        for model in models
        for file in files
    ]


def load_all() -> tuple[list[dict], list[dict]]:
    return load_predictions(RM_FILES), load_predictions(HA_FILES)


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("(empty)")
        return

    columns = ["model", "run"]
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    widths = {
        column: max(len(column), *(len(_format_value(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(_format_value(row.get(column, "")).ljust(widths[column]) for column in columns))


if __name__ == "__main__":
    data_rm, data_ha = load_all()
    print("Reward model predictions:")
    print_table(data_rm)
    print("\nHuman annotation predictions:")
    print_table(data_ha)
