"""Merge RM prompt and PPO unlabeled prompt datasets into a single training set.

Re-numbers the RLAIF ``did`` field (1978+ for the train split, 2880+ for the
test split) and writes the final merged Pillar 4 files:
    data/ppo_unlabeled_prompts_dataset.json
    data/ppo_unlabeled_prompts_dataset_test.json

It also writes the prompt + RLAIF auxiliary files:
    data/ppo_unlabeled_prompts_dataset_prompt.json
    data/ppo_unlabeled_prompts_dataset_prompt_test.json

And the 1k RLAIF-only subset used by the DPO comparison prediction step:
    data/ppo_unlabeled_prompts_dataset_1k.json
    data/ppo_unlabeled_prompts_dataset_1k_test.json

RM prompt inputs are read from their canonical location in
``phase3-rlaif-alignment/reward-model/rm-prompt-dataset/data``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _lib import check_repeated_dids, read_json, write_json

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DATA_DIR = SCRIPT_DIR / "data"
RM_PROMPT_DATA_DIR = SCRIPT_DIR.parents[1] / "reward-model" / "rm-prompt-dataset" / "data"

RLAIF_TRAIN_OFFSET = 1978
RLAIF_TEST_OFFSET = 2880


def _renumber(records: list[dict], start: int) -> list[dict]:
    for idx, entry in enumerate(records, start=start):
        entry["did"] = f"GPT4-{idx:06d}"
    return records


def merge(write_demonstration_prompt_rlaif: bool = False) -> None:
    rlaif_train = _renumber(read_json(LOCAL_DATA_DIR / "ppo_unlabeled_prompts_dataset_original.json"), RLAIF_TRAIN_OFFSET)
    rlaif_test = _renumber(
        read_json(LOCAL_DATA_DIR / "ppo_unlabeled_prompts_dataset_test_original.json"),
        RLAIF_TEST_OFFSET,
    )

    prompt_train = read_json(RM_PROMPT_DATA_DIR / "rm_prompt_dataset_emotional_balanced.json")
    prompt_test = read_json(RM_PROMPT_DATA_DIR / "rm_prompt_dataset_emotional_balanced_test.json")
    rm_prompt_train = read_json(RM_PROMPT_DATA_DIR / "rm_prompt_dataset.json")
    rm_prompt_test = read_json(RM_PROMPT_DATA_DIR / "rm_prompt_dataset_test.json")

    repeated = check_repeated_dids(rm_prompt_train + rm_prompt_test + rlaif_train + rlaif_test)
    if repeated:
        print(f"Repeated 'did' keys found: {repeated}")
    else:
        print("No repeated 'did' keys found.")

    write_json(rlaif_train, LOCAL_DATA_DIR / "ppo_unlabeled_prompts_dataset_1k.json")
    write_json(rlaif_test, LOCAL_DATA_DIR / "ppo_unlabeled_prompts_dataset_1k_test.json")

    prompt_rlaif_train = prompt_train + rlaif_train
    prompt_rlaif_test = prompt_test + rlaif_test
    write_json(prompt_rlaif_train, LOCAL_DATA_DIR / "ppo_unlabeled_prompts_dataset_prompt.json")
    write_json(prompt_rlaif_test, LOCAL_DATA_DIR / "ppo_unlabeled_prompts_dataset_prompt_test.json")

    write_json(
        rm_prompt_train + rlaif_train,
        LOCAL_DATA_DIR / "ppo_unlabeled_prompts_dataset.json",
    )
    write_json(
        rm_prompt_test + rlaif_test,
        LOCAL_DATA_DIR / "ppo_unlabeled_prompts_dataset_test.json",
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
