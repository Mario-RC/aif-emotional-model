"""Parse raw ChatGPT demonstration CSV into structured per-speaker and per-turn CSVs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_CSV = Path("data/sft_demonstration_dataset.csv")
FORMATTED_CSV = Path("data/sft_demonstration_dataset_formatted.csv")
TURNS_CSV = Path("data/sft_demonstration_dataset_turns.csv")

DIALOGUES_PER_ROW = 3
TURNS_PER_DIALOGUE = 4
LINES_PER_TURN = 6
MISSING = "None"
NEUTRAL = "NEUTRAL"

LONG_COLUMNS = [
    "UID", "DID", "SID", "SEG", "EMOTION", "TOPIC", "EXPLANTATION",
    "RESPONSE_1", "RESPONSE_2", "RESPONSE_3",
    "EMOTION_RESPONSE_1", "EMOTION_RESPONSE_2", "EMOTION_RESPONSE_3",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A single Human 1 → Human 2 exchange with structured responses."""

    user: str = MISSING
    user_emotion: str = MISSING
    response_1: str = MISSING
    response_1_emotion: str = MISSING
    response_2: str = MISSING
    response_2_emotion: str = MISSING
    response_3: str = MISSING
    response_3_emotion: str = MISSING
    explanation: str = MISSING


@dataclass
class Dialogue:
    turns: List[Turn] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _extract_tag(line: str) -> tuple[str, str]:
    """Parse a '(EMOTION) text' line into (text, emotion); fall back to MISSING."""
    emo_ini = line.find("(")
    emo_end = line.find(")")
    if emo_ini == -1 or emo_end == -1:
        return MISSING, MISSING
    return line[emo_end + 1:].lstrip(), line[emo_ini + 1:emo_end]


def _extract_step(line: str, label: str) -> str:
    idx = line.find(label)
    if idx == -1:
        return MISSING
    return line[idx + len(label):].lstrip()


class CompletionParser:
    """Parse the free-text COMPLETION from ChatGPT into `Dialogue` objects."""

    def __init__(
        self,
        dialogues_per_row: int = DIALOGUES_PER_ROW,
        turns_per_dialogue: int = TURNS_PER_DIALOGUE,
        lines_per_turn: int = LINES_PER_TURN,
    ) -> None:
        self.dialogues_per_row = dialogues_per_row
        self.turns_per_dialogue = turns_per_dialogue
        self.lines_per_turn = lines_per_turn
        self.markers = [f"Dialogue {i}:" for i in range(1, dialogues_per_row + 1)]

    def parse(self, completion: str) -> List[Dialogue]:
        text = completion.replace("\n\n", "\n").replace("​", "")
        blocks = self._split_dialogue_blocks(text)
        return [self._parse_block(block) for block in blocks if block]

    def _split_dialogue_blocks(self, text: str) -> List[str]:
        positions: List[Optional[int]] = []
        for marker in self.markers:
            idx = text.find(marker)
            positions.append(idx if idx != -1 else None)

        blocks: List[str] = []
        for i, marker in enumerate(self.markers):
            start = positions[i]
            if start is None:
                blocks.append("")
                continue
            end = next(
                (positions[j] for j in range(i + 1, len(self.markers))
                 if positions[j] is not None),
                len(text),
            )
            # +1 skips the trailing newline that usually follows the marker
            blocks.append(text[start + len(marker) + 1:end])
        return blocks

    def _parse_block(self, block: str) -> Dialogue:
        lines = block.lstrip().splitlines()
        dialogue = Dialogue()
        for turn_idx in range(self.turns_per_dialogue):
            chunk = lines[
                turn_idx * self.lines_per_turn:(turn_idx + 1) * self.lines_per_turn
            ]
            if not chunk:
                break
            dialogue.turns.append(self._parse_turn(chunk))
        return dialogue

    @staticmethod
    def _parse_turn(lines: Sequence[str]) -> Turn:
        turn = Turn()
        if len(lines) > 0:
            turn.user, turn.user_emotion = _extract_tag(lines[0])
        if len(lines) > 1:
            turn.response_1, turn.response_1_emotion = _extract_tag(lines[1])
        if len(lines) > 2:
            _, turn.response_2_emotion = _extract_tag(lines[2])
        if len(lines) > 3:
            turn.explanation = _extract_step(lines[3], "Step 2:")
        if len(lines) > 4:
            turn.response_2 = _extract_step(lines[4], "Step 3:")
        if len(lines) > 5:
            turn.response_3, turn.response_3_emotion = _extract_tag(lines[5])
        return turn


# ---------------------------------------------------------------------------
# Long (per-speaker) dataframe builder
# ---------------------------------------------------------------------------

class LongFormatBuilder:
    """Emit one USER row and one CHATBOT row per parsed turn."""

    def __init__(self, speaker_prefix: str = "CHATGPT") -> None:
        self.speaker_prefix = speaker_prefix
        self._rows: List[dict] = []
        self._dialogue_counter = 0

    def _add_completion(self, completion: str, parser: CompletionParser) -> None:
        for dialogue in parser.parse(completion):
            for row in self._iter_flat_turns(dialogue):
                self._rows.append(row)
            self._dialogue_counter += 1

    def _iter_flat_turns(self, dialogue: Dialogue):
        """Yield both USER and CHATBOT rows for each turn, preserving original ordering."""
        turn_counter = 0
        for turn in dialogue.turns:
            yield self._make_user_row(turn, turn_counter)
            turn_counter += 1
            yield self._make_chatbot_row(turn, turn_counter)
            turn_counter += 1

    def _uid(self, turn_idx: int) -> str:
        return (
            f"{self.speaker_prefix}-"
            f"{str(self._dialogue_counter).zfill(6)}-"
            f"{str(turn_idx).zfill(4)}"
        )

    def _make_user_row(self, turn: Turn, turn_idx: int) -> dict:
        uid = self._uid(turn_idx)
        return self._row(
            uid=uid, sid="USER", seg=turn.user,
            emotion=turn.user_emotion, turn=turn,
        )

    def _make_chatbot_row(self, turn: Turn, turn_idx: int) -> dict:
        uid = self._uid(turn_idx)
        return self._row(
            uid=uid, sid="CHATBOT",
            seg=f"{turn.response_1} {turn.response_2} {turn.response_3}",
            emotion=[turn.response_1_emotion, turn.response_2_emotion, turn.response_3_emotion],
            turn=turn,
        )

    def _row(self, uid: str, sid: str, seg: str, emotion, turn: Turn) -> dict:
        return {
            "UID": uid,
            "DID": uid[:-5],
            "SID": sid,
            "SEG": seg,
            "EMOTION": emotion,
            "TOPIC": self._current_topic,
            "EXPLANTATION": turn.explanation,
            "RESPONSE_1": turn.response_1,
            "RESPONSE_2": turn.response_2,
            "RESPONSE_3": turn.response_3,
            "EMOTION_RESPONSE_1": turn.response_1_emotion,
            "EMOTION_RESPONSE_2": turn.response_2_emotion,
            "EMOTION_RESPONSE_3": turn.response_3_emotion,
        }

    def build(self, df_demonstration: pd.DataFrame, parser: CompletionParser) -> pd.DataFrame:
        self._rows = []
        self._dialogue_counter = 0
        for completion, topic in zip(
            df_demonstration["COMPLETION"], df_demonstration["TOPIC"]
        ):
            self._current_topic = topic
            self._add_completion(completion, parser)
        return pd.DataFrame(self._rows, columns=LONG_COLUMNS)


# ---------------------------------------------------------------------------
# Turns (per-turn wide) dataframe builder
# ---------------------------------------------------------------------------

def build_turns_dataframe(df_long: pd.DataFrame) -> pd.DataFrame:
    """Keep only USER rows (even indices) and attach a 'COMPLETION' string column."""
    df_turns = df_long.iloc[::2].reset_index(drop=True)
    df_turns = df_turns.rename(columns={"SEG": "PROMPT"})

    completion = [
        (
            f"({row['EMOTION']}) {row['RESPONSE_1']} "
            f"({row['EMOTION_RESPONSE_2']}) {row['RESPONSE_2']} "
            f"({NEUTRAL}) {row['RESPONSE_3']}"
        )
        for _, row in df_turns.iterrows()
    ]
    df_turns.insert(4, "COMPLETION", completion)
    return df_turns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df_demonstration = pd.read_csv(RAW_CSV, encoding="utf-8")

    parser = CompletionParser()
    builder = LongFormatBuilder()
    df_long = builder.build(df_demonstration, parser)
    df_turns = build_turns_dataframe(df_long)

    FORMATTED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(FORMATTED_CSV, index=False, encoding="utf-8")
    df_turns.to_csv(TURNS_CSV, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
