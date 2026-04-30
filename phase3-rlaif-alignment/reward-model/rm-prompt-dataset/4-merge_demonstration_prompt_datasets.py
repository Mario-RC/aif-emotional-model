"""Merge the RM Prompt Dataset with the SFT Demonstration Dataset into a single training set.

SFT data is read directly from its canonical location in phase 2; no local copy
is materialised here. Re-numbers the demonstration ``did`` field (1000+ for the
train split, 1782+ for the test split) and writes:
    data/rm_prompt_dataset.json
    data/rm_prompt_dataset_test.json
"""

from __future__ import annotations

from _lib import read_json, write_json

SFT_CANONICAL_DIR = "../../../phase2-sft-alignment/sft-model/sft-demonstration-dataset/data"

DEMO_TRAIN_OFFSET = 1000
DEMO_TEST_OFFSET = 1782


def _renumber(records: list[dict], start: int) -> list[dict]:
    for idx, entry in enumerate(records, start=start):
        entry["did"] = f"GPT4-{idx:06d}"
    return records


def merge() -> None:
    demo_train = _renumber(
        read_json(f"{SFT_CANONICAL_DIR}/sft_demonstration_dataset.json"),
        DEMO_TRAIN_OFFSET,
    )
    demo_test = _renumber(
        read_json(f"{SFT_CANONICAL_DIR}/sft_demonstration_dataset_test.json"),
        DEMO_TEST_OFFSET,
    )

    prompt_train = read_json("data/rm_prompt_dataset_emotional_balanced.json")
    prompt_test = read_json("data/rm_prompt_dataset_emotional_balanced_test.json")

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
