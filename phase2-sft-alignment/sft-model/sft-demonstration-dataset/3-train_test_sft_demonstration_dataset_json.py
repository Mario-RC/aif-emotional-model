"""Build LLaMA-Factory-compatible train/test JSON from the per-turn CSV.

Produces an emotion-balanced test set by stratified sampling over
(EMOTION_RESPONSE_1, EMOTION_RESPONSE_2) pairs: for each pair, ``SAMPLES_PER_PAIR``
dialogues are picked, and *all four turns* of every picked dialogue go to test.
The remaining turns form the train+dev split.
"""

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TURNS_CSV = Path("data/sft_demonstration_dataset_turns.csv")
TRAIN_JSON = Path("data/sft_demonstration_dataset.json")
TEST_JSON = Path("data/sft_demonstration_dataset_test.json")
TEST_DIALOGUE_IDS_JSON = Path("data/sft_demonstration_dataset_test_dialogue_ids.json")

TURNS_PER_DIALOGUE = 4
SAMPLES_PER_PAIR = 4
LAST_TURN_UID_SUFFIX = "6"  # UIDs for the chatbot row of the 4th turn end in '6'
NEUTRAL = "NEUTRAL"
DATASET_SET = "sft-demonstration"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = """You are an expert at creating dialogues.

Dialogue and emotional structure:
Human: ({he1}) PROMPT.
Chatbot: ({he1}) RESPONSE_1. ({ce1}) RESPONSE_2. (NEUTRAL) RESPONSE_3.
Human: ({he2}) PROMPT.
Chatbot: ({he2}) RESPONSE_1. ({ce2}) RESPONSE_2. (NEUTRAL) RESPONSE_3.
Human: ({he3}) PROMPT.
Chatbot: ({he3}) RESPONSE_1. ({ce3}) RESPONSE_2. (NEUTRAL) RESPONSE_3.
Human: ({he4}) PROMPT.
Chatbot: ({he4}) RESPONSE_1. ({ce4}) RESPONSE_2. (NEUTRAL) RESPONSE_3.

Dialogue rules:
The response must be open-domain curated. The response should be coherent, empathetic, engaging and proactive.
The chatbot RESPONSE is composed of 3 different sentences (RESPONSE_1, RESPONSE_2 and RESPONSE_3), separated by a period.
Between RESPONSE_1, RESPONSE_2 and RESPONSE_3 should be a max length of 20-25 words.
RESPONSE_3 must be open-ended to follow-up the conversation, so the Human is encouraged to answer with a full long sentence. Avoid yes/no questions.

Emotional response rules:
RESPONSE_1 must contain a {he4} tone.
RESPONSE_2 must contain a {ce4} tone.
RESPONSE_3 must contain a {neutral} tone.

Answer in a single turn to Human. Follow exactly the emotional structure and the emotional and dialogue rules."""


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

class StratifiedEmotionSplitter:
    """Sample N dialogues per (EMOTION_RESPONSE_1, EMOTION_RESPONSE_2) pair."""

    def __init__(
        self,
        turns_per_dialogue: int = TURNS_PER_DIALOGUE,
        samples_per_pair: int = SAMPLES_PER_PAIR,
        last_turn_suffix: str = LAST_TURN_UID_SUFFIX,
        test_dialogue_ids_path: Path | None = TEST_DIALOGUE_IDS_JSON,
        random_state: int | None = None,
    ) -> None:
        self.turns_per_dialogue = turns_per_dialogue
        self.samples_per_pair = samples_per_pair
        self.last_turn_suffix = last_turn_suffix
        self.test_dialogue_ids_path = test_dialogue_ids_path
        self.random_state = random_state

    def split(self, df_turns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.test_dialogue_ids_path and self.test_dialogue_ids_path.exists():
            return self._split_from_dialogue_ids(df_turns, self.test_dialogue_ids_path)

        last_turns = df_turns[df_turns["UID"].str.endswith(self.last_turn_suffix)]
        emotions = list(df_turns["EMOTION_RESPONSE_2"].unique())

        sampled_indices: List[int] = []
        for e1 in emotions:
            for e2 in emotions:
                matching = last_turns[
                    (last_turns["EMOTION_RESPONSE_1"] == e1)
                    & (last_turns["EMOTION_RESPONSE_2"] == e2)
                ]
                if len(matching) < self.samples_per_pair:
                    continue
                sampled_indices.extend(matching.sample(n=self.samples_per_pair, random_state=self.random_state).index)

        self._assert_unique(sampled_indices)

        # Expand each sampled last-turn index into its 4-turn dialogue
        test_indices: List[int] = []
        for idx in sampled_indices:
            test_indices.extend(range(idx - (self.turns_per_dialogue - 1), idx + 1))
        test_indices.sort()

        train_indices = df_turns.index.difference(test_indices).sort_values()

        df_train = df_turns.loc[train_indices].reset_index(drop=True)
        df_test = df_turns.loc[test_indices].reset_index(drop=True)
        return df_train, df_test

    def _split_from_dialogue_ids(self, df_turns: pd.DataFrame, test_dialogue_ids_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        with test_dialogue_ids_path.open(encoding="utf-8") as fh:
            test_dialogue_ids = json.load(fh)

        missing = sorted(set(test_dialogue_ids).difference(df_turns["DIALOGUE_ID"].unique()))
        if missing:
            raise ValueError(f"Test dialogue_id list contains missing values: {missing[:5]}")

        test_indices = df_turns.index[df_turns["DIALOGUE_ID"].isin(test_dialogue_ids)].tolist()
        expected_rows = len(test_dialogue_ids) * self.turns_per_dialogue
        if len(test_indices) != expected_rows:
            raise ValueError(f"Expected {expected_rows} test rows for {len(test_dialogue_ids)} dialogue_id values, found {len(test_indices)}.")

        train_indices = df_turns.index.difference(test_indices).sort_values()
        df_train = df_turns.loc[train_indices].reset_index(drop=True)
        df_test = df_turns.loc[sorted(test_indices)].reset_index(drop=True)
        return df_train, df_test

    @staticmethod
    def _assert_unique(indices: Iterable[int]) -> None:
        indices = list(indices)
        if len(indices) != len(set(indices)):
            raise ValueError("Stratified sampling produced duplicate indices.")


# ---------------------------------------------------------------------------
# LLaMA-Factory export
# ---------------------------------------------------------------------------


class LlamaFactoryExporter:
    """Build LLaMA-Factory `alpaca`-style JSON entries from a per-turn dataframe."""

    def __init__(
        self,
        system_template: str = SYSTEM_TEMPLATE,
        turns_per_dialogue: int = TURNS_PER_DIALOGUE,
        neutral: str = NEUTRAL,
    ) -> None:
        self.system_template = system_template
        self.turns_per_dialogue = turns_per_dialogue
        self.neutral = neutral

    def to_entries(self, df_turns: pd.DataFrame) -> List[dict]:
        entries: List[dict] = []
        for dialogue_id in df_turns["DIALOGUE_ID"].unique():
            df_dialogue = df_turns[df_turns["DIALOGUE_ID"] == dialogue_id].reset_index(drop=True)
            if len(df_dialogue) != self.turns_per_dialogue:
                continue
            entries.append(self._entry_from_dialogue(df_dialogue, dialogue_id))
        return entries

    def _entry_from_dialogue(self, df_dialogue: pd.DataFrame, dialogue_id: str) -> dict:
        prompts = df_dialogue["PROMPT"].tolist()
        completions = df_dialogue["COMPLETION"].tolist()
        human_emotions = df_dialogue["EMOTION"].tolist()
        chatbot_emotions = df_dialogue["EMOTION_RESPONSE_2"].tolist()

        system = self.system_template.format(
            he1=human_emotions[0], ce1=chatbot_emotions[0],
            he2=human_emotions[1], ce2=chatbot_emotions[1],
            he3=human_emotions[2], ce3=chatbot_emotions[2],
            he4=human_emotions[3], ce4=chatbot_emotions[3],
            neutral=self.neutral,
        )

        history = [
            [f"({human_emotions[i]}) {prompts[i]}", completions[i]]
            for i in range(self.turns_per_dialogue - 1)
        ]
        instruction = f"({human_emotions[-1]}) {prompts[-1]}"

        return {
            "system": system,
            "history": history,
            "instruction": instruction,
            "input": "",
            "output": completions[-1],
            "dialogue_id": dialogue_id,
            "set": DATASET_SET,
        }

    @staticmethod
    def save(entries: List[dict], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(entries, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df_turns = pd.read_csv(TURNS_CSV, encoding="utf-8")

    splitter = StratifiedEmotionSplitter()
    df_train, df_test = splitter.split(df_turns)

    exporter = LlamaFactoryExporter()
    exporter.save(exporter.to_entries(df_train), TRAIN_JSON)
    exporter.save(exporter.to_entries(df_test), TEST_JSON)


if __name__ == "__main__":
    main()
