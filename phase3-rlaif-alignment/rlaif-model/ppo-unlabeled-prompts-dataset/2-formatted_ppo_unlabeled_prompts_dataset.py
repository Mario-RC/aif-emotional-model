"""Parse the raw GPT-4O completions into a flat per-turn DataFrame.

Inputs:
    data/ppo_unlabeled_prompts_dataset_completions.csv
Outputs:
    data/ppo_unlabeled_prompts_dataset_formatted.csv  -- per-segment (USER/CHATBOT)
    data/ppo_unlabeled_prompts_dataset_turns.csv      -- per-user-turn (with COMPLETION)

Each completion contains a single 4-turn dialogue (in contrast to the
RM Prompt Dataset variant, which packs two dialogues per completion).
"""

from __future__ import annotations

import argparse

import pandas as pd

DEFAULT_INPUT = "data/ppo_unlabeled_prompts_dataset_completions.csv"
DEFAULT_FORMATTED = "data/ppo_unlabeled_prompts_dataset_formatted.csv"
DEFAULT_TURNS = "data/ppo_unlabeled_prompts_dataset_turns.csv"


def _split_turn(turn_lines: list[str]) -> dict | None:
    """Decode the 6-line turn block into per-field strings."""
    if not turn_lines:
        return None

    def emo_split(line: str) -> tuple[str, str]:
        emo_ini = line.find("(")
        emo_end = line.find(")")
        return line[emo_ini + 1 : emo_end], line[emo_end + 1 :].lstrip()

    out = {
        "user": "None", "user_emotion": "None",
        "chatbot_r1": "None", "chatbot_r1_emotion": "None",
        "chatbot_r2": "None", "chatbot_r2_emotion": "None",
        "chatbot_r3": "None", "chatbot_r3_emotion": "None",
        "explanation_r2": "None",
    }

    try:
        out["user_emotion"], out["user"] = emo_split(turn_lines[0])
    except Exception:
        pass
    try:
        out["chatbot_r1_emotion"], out["chatbot_r1"] = emo_split(turn_lines[1])
    except Exception:
        pass
    try:
        out["chatbot_r2_emotion"], _ = emo_split(turn_lines[2])
    except Exception:
        pass
    try:
        step2 = turn_lines[3].find("Step 2:")
        out["explanation_r2"] = turn_lines[3][step2 + 7 :].lstrip()
    except Exception:
        pass
    try:
        step3 = turn_lines[4].find("Step 3: ")
        out["chatbot_r2"] = turn_lines[4][step3 + 7 :].lstrip()
    except Exception:
        pass
    try:
        out["chatbot_r3_emotion"], out["chatbot_r3"] = emo_split(turn_lines[5])
    except Exception:
        pass
    return out


def _parse_dialogue(completion: str) -> tuple[list, list, list] | None:
    """Parse a single GPT completion (one 4-turn dialogue, 6 lines per turn)."""
    completion = completion.replace("\n\n", "\n").replace("​", "")
    lines = completion.lstrip().splitlines()
    turns = [lines[i * 6 : (i + 1) * 6] for i in range(4)]

    dialogue, explanations, emotions = [], [], []
    for turn_lines in turns:
        decoded = _split_turn(turn_lines)
        if decoded is None:
            return None
        dialogue.append([decoded["user"], decoded["chatbot_r1"], decoded["chatbot_r2"], decoded["chatbot_r3"]])
        explanations.append(decoded["explanation_r2"])
        emotions.append([
            decoded["user_emotion"],
            decoded["chatbot_r1_emotion"],
            decoded["chatbot_r2_emotion"],
            decoded["chatbot_r3_emotion"],
        ])
    return dialogue, explanations, emotions


def _parse_completions(df: pd.DataFrame) -> tuple[list, list, list]:
    dialogues, explanations, emotions = [], [], []
    for completion in df["COMPLETION"]:
        parsed = _parse_dialogue(completion)
        if parsed is None:
            break
        dialogues.append(parsed[0])
        explanations.append(parsed[1])
        emotions.append(parsed[2])
    return dialogues, explanations, emotions


def _build_segment_dataframe(
    dialogues: list, explanations: list, emotions: list, topics: pd.Series
) -> pd.DataFrame:
    """Expand parsed dialogues into a long-format USER/CHATBOT segment dataframe."""
    rows = {
        key: []
        for key in [
            "uid", "did", "sid", "seg", "emotion_list", "explanation_list",
            "r1", "r2", "r3", "r1_emotion", "r2_emotion", "r3_emotion", "topic_list",
        ]
    }

    for dial_count, (dialogue, explanation, emotion, topic) in enumerate(
        zip(dialogues, explanations, emotions, topics)
    ):
        for turn_count, (dial, expl, emo) in enumerate(zip(dialogue, explanation, emotion)):
            base_uid = f"GPT4-{dial_count:06d}"
            # USER segment
            rows["uid"].append(f"{base_uid}-{turn_count * 2:04d}")
            rows["did"].append(base_uid)
            rows["sid"].append("USER")
            rows["seg"].append(dial[0])
            rows["r1"].append(dial[1])
            rows["r2"].append(dial[2])
            rows["r3"].append(dial[3])
            rows["r1_emotion"].append(emo[1])
            rows["r2_emotion"].append(emo[2])
            rows["r3_emotion"].append(emo[3])
            rows["emotion_list"].append(emo[0])
            rows["topic_list"].append(topic)
            rows["explanation_list"].append(expl)

            # CHATBOT segment
            rows["uid"].append(f"{base_uid}-{turn_count * 2 + 1:04d}")
            rows["did"].append(base_uid)
            rows["sid"].append("CHATBOT")
            rows["seg"].append(f"{dial[1]} {dial[2]} {dial[3]}")
            rows["r1"].append(dial[1])
            rows["r2"].append(dial[2])
            rows["r3"].append(dial[3])
            rows["r1_emotion"].append(emo[1])
            rows["r2_emotion"].append(emo[2])
            rows["r3_emotion"].append(emo[3])
            rows["emotion_list"].append(emo[1:])
            rows["topic_list"].append(topic)
            rows["explanation_list"].append(expl)

    return pd.DataFrame(
        list(zip(
            rows["uid"], rows["did"], rows["sid"], rows["seg"], rows["emotion_list"],
            rows["explanation_list"], rows["r1"], rows["r2"], rows["r3"],
            rows["r1_emotion"], rows["r2_emotion"], rows["r3_emotion"], rows["topic_list"],
        )),
        columns=[
            "UID", "DID", "SID", "SEG", "EMOTION_SEG", "EXPLANTATION",
            "RESPONSE_1", "RESPONSE_2", "RESPONSE_3",
            "EMOTION_RESPONSE_1", "EMOTION_RESPONSE_2", "EMOTION_RESPONSE_3", "TOPIC",
        ],
    )


def _build_turns_dataframe(df_segments: pd.DataFrame) -> pd.DataFrame:
    """Take the USER rows only and add the per-turn ``COMPLETION`` (emotion-wrapped) field."""
    df_turns = df_segments.iloc[::2].reset_index(drop=True).rename(
        columns={"SEG": "PROMPT", "EMOTION_SEG": "EMOTION_PROMPT"}
    )
    completions = [
        f"({row['EMOTION_PROMPT']}) {row['RESPONSE_1']} ({row['EMOTION_RESPONSE_2']}) {row['RESPONSE_2']} (NEUTRAL) {row['RESPONSE_3']}"
        for _, row in df_turns.iterrows()
    ]
    df_turns.insert(5, "COMPLETION", completions)
    return df_turns


def format_ppo_unlabeled_prompts_dataset(
    in_csv: str = DEFAULT_INPUT,
    formatted_csv: str = DEFAULT_FORMATTED,
    turns_csv: str = DEFAULT_TURNS,
) -> None:
    df = pd.read_csv(in_csv, encoding="utf-8")
    dialogues, explanations, emotions = _parse_completions(df)
    df_segments = _build_segment_dataframe(dialogues, explanations, emotions, df["TOPIC"])
    df_turns = _build_turns_dataframe(df_segments)

    df_segments.to_csv(formatted_csv, index=False, encoding="utf-8")
    df_turns.to_csv(turns_csv, index=False, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-csv", default=DEFAULT_INPUT)
    parser.add_argument("--formatted-csv", default=DEFAULT_FORMATTED)
    parser.add_argument("--turns-csv", default=DEFAULT_TURNS)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    format_ppo_unlabeled_prompts_dataset(
        in_csv=args.in_csv,
        formatted_csv=args.formatted_csv,
        turns_csv=args.turns_csv,
    )
