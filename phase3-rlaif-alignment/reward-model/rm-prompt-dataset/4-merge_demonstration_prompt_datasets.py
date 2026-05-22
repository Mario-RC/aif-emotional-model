"""Merge the RM Prompt Dataset with the SFT Demonstration Dataset into a single training set.

SFT data is read directly from its canonical location in phase 2; no local copy
is materialised here. Final JSON records keep ``dialogue_id`` as the only
dialogue identity field and write:
    data/rm_prompt_dataset.json
    data/rm_prompt_dataset_test.json
"""

from __future__ import annotations


from _lib import read_json, write_json

SFT_CANONICAL_DIR = "../../../phase2-sft-alignment/sft-model/sft-demonstration-dataset/data"



def _prepare_records(records: list[dict], expected_set: str) -> list[dict]:
    for entry in records:
        if not entry.get("dialogue_id"):
            raise ValueError("Record is missing dialogue_id.")
        entry.setdefault("set", expected_set)
        if entry["set"] != expected_set:
            raise ValueError(f"Expected set={expected_set!r}, found {entry['set']!r}.")
    return records


def merge() -> None:
    demo_train = _prepare_records(read_json(f"{SFT_CANONICAL_DIR}/sft_demonstration_dataset.json"), "sft-demonstration")
    demo_test = _prepare_records(read_json(f"{SFT_CANONICAL_DIR}/sft_demonstration_dataset_test.json"), "sft-demonstration")

    prompt_train = _prepare_records(read_json("data/rm_prompt_dataset_emotional_balanced.json"), "rm-prompt")
    prompt_test = _prepare_records(read_json("data/rm_prompt_dataset_emotional_balanced_test.json"), "rm-prompt")

    write_json(
        prompt_train + demo_train,
        "data/rm_prompt_dataset.json",
    )
    write_json(
        prompt_test + demo_test,
        "data/rm_prompt_dataset_test.json",
    )


if __name__ == "__main__":
    merge()
