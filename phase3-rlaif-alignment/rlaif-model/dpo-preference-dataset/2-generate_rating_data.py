"""Generate per-LLM rating CSVs for each DPO Preference Dataset entry.

Produces ``data/<LLM>/dpo_preference_dataset_rate_<LLM>[_test].csv``. Checkpoints are
written every 10 entries under ``data/<LLM>/records/``.

Usage::

    python 2-generate_rating_data.py --llm GPT-4O
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from _lib import (
    DEFAULT_LLM_CONFIG_FILE,
    RatingClient,
    build_rating_prompt,
    get_emotional_definitions,
    get_emotions_table,
    load_llm_config,
    rating_rules,
    read_json,
    split_emo_utt,
    with_suffix,
)

logging.basicConfig(level=logging.INFO)
warnings.simplefilter("ignore")

CHECKPOINT_INTERVAL = 10
DATA_COLUMNS = ["DID", "PROMPT", "COMPLETION", "EMOTIONS", "EXPRESSION_LEVEL", "MODIFIED"]


def _iter_records(
    client: RatingClient, dpo_preference_dataset: Iterable[dict]
) -> Iterable[tuple[str, str, str, list[str], list[int], bool]]:
    for entry in tqdm(dpo_preference_dataset):
        human_emo, chatbot_emo = split_emo_utt(entry["prompt"], entry["target"])
        get_emotional_definitions(human_emo, chatbot_emo)
        empathy_level, emotion_level, question_level = get_emotions_table(human_emo, chatbot_emo)
        empathy_rule, emotion_rule, question_rule = rating_rules(human_emo, chatbot_emo)

        modified = entry.get("predict_sft_modified_label", "None") != "None"
        prompt = build_rating_prompt(
            entry, empathy_rule, emotion_rule, question_rule, modified=modified
        )
        prompt_message = "\n\n".join(prompt)
        completion = client.complete(prompt)

        yield (
            entry["did"],
            prompt_message,
            completion,
            [human_emo, chatbot_emo],
            [empathy_level, emotion_level, question_level],
            modified,
        )


def generate_rating_data(
    llm_name: str,
    is_test: bool = False,
    config_file: str = DEFAULT_LLM_CONFIG_FILE,
    start_offset: int = 0,
) -> None:
    cfg = load_llm_config(llm_name, config_file)
    print(cfg)
    client = RatingClient(llm_name, cfg)

    input_file = f"data/{with_suffix('dpo_preference_dataset', 'json', is_test)}"
    dpo_preference_dataset = read_json(input_file)[start_offset:]

    out_dir = f"data/{llm_name}"
    records_dir = f"{out_dir}/records"
    os.makedirs(records_dir, exist_ok=True)

    rows: list[tuple] = []
    chkpt = 0
    for idx, record in enumerate(_iter_records(client, dpo_preference_dataset), start=1):
        rows.append((record[0], record[1], [record[2]], record[3], record[4], record[5]))
        if idx % CHECKPOINT_INTERVAL == 0:
            df = pd.DataFrame(rows, columns=DATA_COLUMNS)
            chk_name = f"dpo_preference_dataset_rate_{llm_name}_checkpoint_{chkpt}{'_test' if is_test else ''}.csv"
            df.to_csv(f"{records_dir}/{chk_name}", index=False)
            chkpt += 1

    df = pd.DataFrame(rows, columns=DATA_COLUMNS)
    out_name = with_suffix(f"dpo_preference_dataset_rate_{llm_name}", "csv", is_test)
    df.to_csv(f"{out_dir}/{out_name}", index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm",
        default=os.environ.get("LLM", "GPT-4O"),
        help="LLM key in config_llm.json (e.g. GPT-4O, CLAUDE-3.5-SONNET, GEMINI-1.5-PRO, LLAMA-3.1-405B).",
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--config", default=DEFAULT_LLM_CONFIG_FILE)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_rating_data(
        llm_name=args.llm,
        is_test=args.test,
        config_file=args.config,
        start_offset=args.start,
    )
