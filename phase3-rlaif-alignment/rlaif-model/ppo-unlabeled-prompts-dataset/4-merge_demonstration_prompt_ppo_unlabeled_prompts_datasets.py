"""Merge RM prompt and PPO unlabeled prompt datasets into a single training set.

Re-numbers the PPO unlabeled prompt ``did`` field (1978+ for the train split,
2880+ for the test split) and writes the final merged Pillar 4 files:
    data/ppo_unlabeled_prompts_dataset.json
    data/ppo_unlabeled_prompts_dataset_test.json

It also writes the prompt + PPO auxiliary files:
    data/ppo_unlabeled_prompts_dataset_prompt.json
    data/ppo_unlabeled_prompts_dataset_prompt_test.json

And the 1k PPO-only subset used by the DPO comparison prediction step:
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

PPO_TRAIN_OFFSET = 1978
PPO_TEST_OFFSET = 2880


def _renumber(records: list[dict], start: int) -> list[dict]:
    for idx, entry in enumerate(records, start=start):
        entry["did"] = f"GPT4-{idx:06d}"
    return records


def merge(_deprecated_write_combined: bool = False) -> None:
    ppo_train = _renumber(read_json(LOCAL_DATA_DIR / "ppo_unlabeled_prompts_dataset_original.json"), PPO_TRAIN_OFFSET)
    ppo_test = _renumber(
        read_json(LOCAL_DATA_DIR / "ppo_unlabeled_prompts_dataset_test_original.json"),
        PPO_TEST_OFFSET,
    )

    prompt_train = read_json(RM_PROMPT_DATA_DIR / "rm_prompt_dataset_emotional_balanced.json")
    prompt_test = read_json(RM_PROMPT_DATA_DIR / "rm_prompt_dataset_emotional_balanced_test.json")
    rm_prompt_train = read_json(RM_PROMPT_DATA_DIR / "rm_prompt_dataset.json")
    rm_prompt_test = read_json(RM_PROMPT_DATA_DIR / "rm_prompt_dataset_test.json")

    repeated = check_repeated_dids(rm_prompt_train + rm_prompt_test + ppo_train + ppo_test)
    if repeated:
        print(f"Repeated 'did' keys found: {repeated}")
    else:
        print("No repeated 'did' keys found.")

    write_json(ppo_train, LOCAL_DATA_DIR / "ppo_unlabeled_prompts_dataset_1k.json")
    write_json(ppo_test, LOCAL_DATA_DIR / "ppo_unlabeled_prompts_dataset_1k_test.json")

    prompt_ppo_train = prompt_train + ppo_train
    prompt_ppo_test = prompt_test + ppo_test
    write_json(prompt_ppo_train, LOCAL_DATA_DIR / "ppo_unlabeled_prompts_dataset_prompt.json")
    write_json(prompt_ppo_test, LOCAL_DATA_DIR / "ppo_unlabeled_prompts_dataset_prompt_test.json")

    write_json(
        rm_prompt_train + ppo_train,
        LOCAL_DATA_DIR / "ppo_unlabeled_prompts_dataset.json",
    )
    write_json(
        rm_prompt_test + ppo_test,
        LOCAL_DATA_DIR / "ppo_unlabeled_prompts_dataset_test.json",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-demonstration-prompt-rlaif",
        dest="deprecated_write_combined",
        action="store_true",
        help="Deprecated; final prompt + PPO outputs are always written to data/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    merge(_deprecated_write_combined=args.deprecated_write_combined)
