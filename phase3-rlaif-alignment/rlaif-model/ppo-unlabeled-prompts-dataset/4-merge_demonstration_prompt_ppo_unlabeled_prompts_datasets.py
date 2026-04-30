"""Merge demonstration, prompt and RLAIF datasets into a single training set.

Re-numbers the RLAIF ``did`` field (1978+ for the train split, 2880+ for the
test split) and writes the final merged Pillar 4 files:
    data/ppo_unlabeled_prompts_dataset.json
    data/ppo_unlabeled_prompts_dataset_test.json

It also writes the prompt + RLAIF auxiliary files:
    data/ppo_unlabeled_prompts_dataset_prompt.json
    data/ppo_unlabeled_prompts_dataset_prompt_test.json

The SFT demonstration, RM prompt and RLAIF inputs all live directly in ``data/``.
"""

from __future__ import annotations

import argparse

from _lib import check_repeated_dids, read_json, write_json

RLAIF_TRAIN_OFFSET = 1978
RLAIF_TEST_OFFSET = 2880


def _renumber(records: list[dict], start: int) -> list[dict]:
    for idx, entry in enumerate(records, start=start):
        entry["did"] = f"GPT4-{idx:06d}"
    return records


def merge(write_demonstration_prompt_rlaif: bool = False) -> None:
    rlaif_train = _renumber(read_json("data/ppo_unlabeled_prompts_dataset_original.json"), RLAIF_TRAIN_OFFSET)
    rlaif_test = _renumber(read_json("data/ppo_unlabeled_prompts_dataset_test_original.json"), RLAIF_TEST_OFFSET)

    prompt_train = read_json("data/prompt_data_emotional_balanced.json")
    prompt_test = read_json("data/prompt_data_emotional_balanced_test.json")
    _sft_train = read_json("data/sft_demonstration_dataset.json")
    _sft_test = read_json("data/sft_demonstration_dataset_test.json")
    rm_prompt_train = read_json("data/rm_prompt_dataset.json")
    rm_prompt_test = read_json("data/rm_prompt_dataset_test.json")

    repeated = check_repeated_dids(rm_prompt_train + rm_prompt_test + rlaif_train + rlaif_test)
    if repeated:
        print(f"Repeated 'did' keys found: {repeated}")
    else:
        print("No repeated 'did' keys found.")

    prompt_rlaif_train = prompt_train + rlaif_train
    prompt_rlaif_test = prompt_test + rlaif_test
    write_json(prompt_rlaif_train, "data/ppo_unlabeled_prompts_dataset_prompt.json")
    write_json(prompt_rlaif_test, "data/ppo_unlabeled_prompts_dataset_prompt_test.json")

    write_json(
        rm_prompt_train + rlaif_train,
        "data/ppo_unlabeled_prompts_dataset.json",
    )
    write_json(
        rm_prompt_test + rlaif_test,
        "data/ppo_unlabeled_prompts_dataset_test.json",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-demonstration-prompt-rlaif",
        action="store_true",
        help="Deprecated; final demonstration + prompt + RLAIF outputs are always written to data/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    merge(write_demonstration_prompt_rlaif=args.write_demonstration_prompt_rlaif)
