"""Generate the seed PPO Unlabeled Prompts Dataset CSVs by querying Azure-OpenAI for emotional dialogues.

For each topic in :data:`_lib.TOPICS` (1000 topics) ask the LLM to produce a 4-turn
dialogue with the requested emotion sequence. Checkpoints are written every 30
entries to ``data/records/ppo_unlabeled_prompts_dataset_checkpoint_<n>.csv`` and the final
result lands in ``data/ppo_unlabeled_prompts_dataset_completions.csv``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings

import pandas as pd
from openai import AzureOpenAI
from tqdm import tqdm

from _lib import build_topics_emotions

logging.basicConfig(level=logging.INFO)
warnings.simplefilter("ignore")

DEFAULT_GPT_KEY = "GPT-4O"

CHECKPOINT_INTERVAL = 30

SYSTEM_PROMPT = "You are an expert at creating emotional dialogues."

INSTRUCTIONS_PROMPT = """Dialogue and emotional structure:
Turn 1 Human 1: (prompt_emotion_1) PROMPT.
Turn 1 Human 2: (prompt_emotion_1) RESPONSE_1.
(response_emotion_1) Let's think step by step. Step 1: The response must contain an emotional tone of response_emotion_1.
Step 2: Human 2 should express
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.
Turn 2 Human 1: (prompt_emotion_2) PROMPT.
Turn 2 Human 2: (prompt_emotion_2) RESPONSE_1.
(response_emotion_2) Let's think step by step. Step 1: The response must contain an emotional tone of response_emotion_2.
Step 2: Human 2 should express
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.
Turn 3 Human 1: (prompt_emotion_3) PROMPT.
Turn 3 Human 2: (prompt_emotion_3) RESPONSE_1.
(response_emotion_3) Let's think step by step. Step 1: The response must contain an emotional tone of response_emotion_3.
Step 2: Human 2 should express
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.
Turn 4 Human 1: (prompt_emotion_4) PROMPT.
Turn 4 Human 2: (prompt_emotion_4) RESPONSE_1.
(response_emotion_4) Let's think step by step. Step 1: The response must contain an emotional tone of response_emotion_4.
Step 2: Human 2 should express
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.

Dialogue rules:
Create an open-domain curated dialogue between two humans. The dialogue should be coherent, engaging and proactive.
Between RESPONSE_1, RESPONSE_2 and RESPONSE_3 must be a max length of 20-25 words.
RESPONSE_3 must be open-ended to follow-up the conversation, so the Human 1 is encouraged to answer with a full long sentence. Avoid yes/no questions.

Emotional rules:
Remember these different key_emotions emotions: ANGER, DISGUST(DISPLEASURE/DISLIKE), FEAR, HAPPINESS, SADNESS, SURPRISE and NEUTRAL.
For prompt_emotion and response_emotion randomly select different emotions among the key_emotions at each turn.
PROMPT and RESPONSE_1 must contain the same prompt_emotion tone and be empathetic.
RESPONSE_2 in each turn of Human 2 must show a lot of emotion and not be at all empathetic to Human 1.
RESPONSE_3 must contain the NEUTRAL tone.

Emotional definitions:
ANGER: a strong feeling of annoyance, displeasure, or hostility. Example: "she could barely restrain her anger at this comment".
DISGUST: a feeling of revulsion, aversion or strong disapproval aroused by something unpleasant or offensive. Example: "the sight filled her with disgust".
FEAR: an unpleasant emotion caused by the belief that someone or something is dangerous, likely to cause pain, or a threat. Example: "farmers fear that they will lose business".
HAPPINESS: the state of being happy. Example: "she struggled to find happiness in her life".
SADNESS: the condition or quality of being sad. Example: "a source of great sadness".
SURPRISE: an unexpected or astonishing event, fact, or thing. Example: "the announcement was a complete surprise".
NEUTRAL: having no strongly marked or positive characteristics or features. Example: "the tone was neutral, devoid of sentiment"."""

EXAMPLE_PROMPT = """Dialogue example:
Turn 1 Human 1: (HAPPINESS) Guess what, I just got engaged!
Turn 1 Human 2: (HAPPINESS) Congratulations! That's wonderful news!
(FEAR) Let's think step by step. Step 1: The response must contain an emotional tone of FEAR.
Step 2: Human 2 must express terror of commitment.
Step 3: I'm so happy for you, but I must admit, wedding planning sounds terrifying.
(NEUTRAL) Have you set a date yet?
Turn 2 Human 1: (HAPPINESS) We're thinking about next summer, and it's so exciting, even though we haven't set a date yet.
Turn 2 Human 2: (HAPPINESS) Summer weddings are beautiful!
(NEUTRAL) Let's think step by step. Step 1: The response must contain an emotional tone of NEUTRAL.
Step 2: Human 2 must express neutrality regarding the wedding organization, without expressing any emotional tone.
Step 3: Although, wedding planning can be challenging, try to maintain a calm approach.
(NEUTRAL) Have you picked a venue yet?
Turn 3 Human 1: (SADNESS) It's a bit overwhelming. We've started browsing, but there are so many options, we have thought about a wedding planner.
Turn 3 Human 2: (SADNESS) I completely understand the overwhelm.
(SURPRISE) Let's think step by step. Step 1: The response must contain an emotional tone of SURPRISE.
Step 2: Human 2 should express surprise about the wedding planner.
Step 3: That's a fantastic idea! I hadn't even thought about hiring a wedding planner.
(NEUTRAL) Do you have any ideas for the theme or venue?
Turn 4 Human 1: (DISGUST) We're leaning towards a beach wedding, but they're too expensive.
Turn 4 Human 2: (DISGUST) It's true, beach weddings can get pricey.
(DISGUST) Let's think step by step. Step 1: The response must contain an emotional tone of DISGUST.
Step 2: Human 2 should express displeasure at the cost of the wedding.
Step 3: It's normal for the cost to put you off.
(NEUTRAL) Have you considered other scenic outdoor options that might be more budget-friendly?"""


def _emotion_block(emotions: list[tuple[str, str]]) -> str:
    """Build the per-dialogue ``(emotion_a) PROMPT`` template block (single dialogue, 4 turns)."""
    lines: list[str] = []
    for turn_idx in range(4):
        he, ce = emotions[turn_idx]
        lines.extend([
            f"Turn {turn_idx + 1} Human 1: ({he}) PROMPT.",
            f"Turn {turn_idx + 1} Human 2: ({he}) RESPONSE_1.",
            f"({ce}) Let's think step by step. Step 1: The response must contain an emotional tone of {ce}.",
            "Step 2: Human 2 should express ",
            "Step 3: RESPONSE_2.",
            "(NEUTRAL) RESPONSE_3.",
        ])
    return "\n".join(lines)


def make_messages(topic: str, emotion: list[tuple[str, str]]) -> list[dict]:
    """Build the four-message chat prompt for a single topic."""
    dialogue_emotions = (
        "Follow exactly all dialogues and emotional rules, emotional definitions, emotional structure "
        "and the dialogue example.\n"
        "IMPORTANT: Pay special attention to show steps 1, 2 and 3 in each of the turns.\n"
        f"Provide a new different dialogues following the template below where the topic of the conversations is {topic}.\n\n"
        f"{_emotion_block(emotion)}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTRUCTIONS_PROMPT},
        {"role": "assistant", "content": EXAMPLE_PROMPT},
        {"role": "user", "content": dialogue_emotions},
    ]


def _resolve_secret(cfg: dict, key: str, env_var: str, placeholder: str) -> str:
    """Return the cfg value for ``key`` if it is set to a real value;
    otherwise fall back to the environment variable ``env_var``.

    A value is considered "not real" when it is empty or still wrapped in
    angle brackets (the sanitization placeholder convention).
    """
    value = cfg.get(key, "") or ""
    if not value or (value.startswith("<") and value.endswith(">")):
        return os.getenv(env_var, placeholder)
    return value


def build_client(cfg: dict) -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=_resolve_secret(
            cfg, "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_ENDPOINT", "<YOUR_AZURE_ENDPOINT_HERE>"
        ),
        api_key=_resolve_secret(
            cfg, "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "<YOUR_AZURE_OPENAI_API_KEY_HERE>"
        ),
        api_version=cfg["OPENAI_API_VERSION"],
    )


def generate_ppo_unlabeled_prompts_dataset(
    gpt_key: str = DEFAULT_GPT_KEY,
    config_file: str = "config_gpt.json",
    out_csv: str = "data/ppo_unlabeled_prompts_dataset_completions.csv",
    sleep_s: float = 1.0,
    dry_run: bool = False,
) -> None:
    with open(config_file) as f:
        cfg = json.load(f)[gpt_key]
    print(cfg)

    client = build_client(cfg)
    topics_emotions = build_topics_emotions()

    dids, prompts, completions, topic_list, emotions_list = [], [], [], [], []
    chkpt = 0

    for idx, (topic, emotion) in enumerate(tqdm(topics_emotions, start=1), start=1):
        time.sleep(sleep_s)
        messages = make_messages(topic, emotion)

        if dry_run:
            completion = ""
        else:
            response = client.chat.completions.create(model=cfg["MODEL"], messages=messages)
            completion = response.choices[0].message.content

        dids.append(f"GPT4-{idx - 1:04d}")
        prompts.append(messages)
        completions.append(completion)
        topic_list.append(topic)
        emotions_list.append(emotion)

        if idx % CHECKPOINT_INTERVAL == 0:
            df_chk = pd.DataFrame(
                list(zip(dids, prompts, completions, topic_list, emotions_list)),
                columns=["DID", "PROMPT", "COMPLETION", "TOPIC", "EMOTIONS"],
            )
            df_chk.to_csv(f"data/records/ppo_unlabeled_prompts_dataset_checkpoint_{chkpt}.csv", index=False)
            chkpt += 1

    df = pd.DataFrame(
        list(zip(dids, prompts, completions, topic_list, emotions_list)),
        columns=["DID", "PROMPT", "COMPLETION", "TOPIC", "EMOTIONS"],
    )
    df.to_csv(out_csv, index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt", default=DEFAULT_GPT_KEY)
    parser.add_argument("--config", default="config_gpt.json")
    parser.add_argument("--out", default="data/ppo_unlabeled_prompts_dataset_completions.csv")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_ppo_unlabeled_prompts_dataset(
        gpt_key=args.gpt,
        config_file=args.config,
        out_csv=args.out,
        dry_run=args.dry_run,
    )
