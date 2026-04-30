"""Strip ``predict_sft_modified`` from the filtered RLAIF data and label every row "None".

Unlike the corresponding script in ``phase3-rlaif-alignment/reward-model/rm-comparison-dataset/candidate-curation/``,
the RLAIF variant does not actually inject negative LLM-rewritten samples — it
simply renames the upstream ``dpo_comparison_dataset_filtered[*test].json`` into
``dpo_comparison_dataset[*test].json`` after dropping the unused field.
"""

from __future__ import annotations

import argparse

from _lib import read_json, with_suffix, write_json


def add_negative_samples(is_test: bool = False) -> None:
    in_path = f"data/{with_suffix('dpo_comparison_dataset_filtered', 'json', is_test)}"
    out_path = f"data/{with_suffix('dpo_comparison_dataset', 'json', is_test)}"

    data = read_json(in_path)
    for entry in data:
        entry.pop("predict_sft_modified", None)
        entry["predict_sft_modified_label"] = "None"

    write_json(data, out_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    add_negative_samples(is_test=args.test)
