"""Post-process the per-row RANK into pairwise winner/loser comparisons + responses."""

from __future__ import annotations

import argparse
import ast
import json
import re

import pandas as pd

from _lib import create_pairwise_comparisons, with_suffix

IDENTITY_COLUMN = "DIALOGUE_ID"
DEFAULT_SYSTEM = "You are an expert at creating dialogues."


def _add_extra_value_for_modified(row: pd.Series) -> pd.Series:
    """For entries flagged as ``MODIFIED``, append a sentinel rank to mark the modified response."""
    if row["MODIFIED"]:
        rank_list = ast.literal_eval(row["RANK"])
        rank_list.append(max(rank_list) + 1)
        row["RANK"] = rank_list
    return row


def _identity_column(df: pd.DataFrame) -> str:
    if IDENTITY_COLUMN in df.columns:
        return IDENTITY_COLUMN
    raise KeyError(f"Expected {IDENTITY_COLUMN} column.")


def _series_value(row: pd.Series, key: str, default: str = "") -> str:
    if key not in row:
        return default
    value = row[key]
    if pd.isna(value):
        return default
    return value


def _preference_id_prefix(identity: str) -> str:
    text = str(identity)
    if "-" in text:
        _, suffix = text.split("-", 1)
        return f"RLAIFEP-{suffix}"
    return f"RLAIFEP-{text}"


def _validate_unique_rank_ids(df: pd.DataFrame, id_col: str) -> None:
    ids = df[id_col].astype(str)
    duplicated = ids[ids.duplicated()].drop_duplicates().tolist()
    if duplicated:
        raise ValueError(
            f"Rank outputs contain duplicated dialogue IDs: {duplicated[:5]}"
        )


def add_modified_rank(is_test: bool) -> pd.DataFrame:
    in_path = f"data/{with_suffix('rm_preference_dataset_models_results_rank', 'csv', is_test)}"
    out_path = f"data/{with_suffix('rm_preference_dataset_models_results_rank_modified', 'csv', is_test)}"

    df = pd.read_csv(in_path)
    _validate_unique_rank_ids(df, _identity_column(df))
    df = df.apply(_add_extra_value_for_modified, axis=1)
    df.to_csv(out_path, index=False)
    return df


def build_pairwise_dataframe(is_test: bool) -> pd.DataFrame:
    in_path = f"data/{with_suffix('rm_preference_dataset_models_results_rank_modified', 'csv', is_test)}"
    df = pd.read_csv(in_path)
    id_col = _identity_column(df)

    rows: list[dict] = []
    for _, row in df.iterrows():
        ranking = ast.literal_eval(row["RANK"])
        for winner, loser in create_pairwise_comparisons(ranking):
            rows.append({IDENTITY_COLUMN: row[id_col], "WINNER": winner, "LOSER": loser})
    pairwise_df = pd.DataFrame(rows)
    pairwise_df.to_csv(f"data/{with_suffix('rm_preference_dataset_df', 'csv', is_test)}", index=False)
    return pairwise_df


def _human_emotion(text: str) -> str:
    match = re.match(r"\(([^)]+)\)", str(text).strip())
    if not match:
        raise ValueError(f"Could not extract human emotion from {text!r}.")
    return match.group(1)


def _restore_response_tags(response: str, human_utterance: str) -> str:
    """Invert the paid rating prompt tags back to dataset-style emotion tags."""
    restored = re.sub(
        r"^\(EMPATHY\)",
        f"({_human_emotion(human_utterance)})",
        str(response).strip(),
        count=1,
    )
    return restored.replace("(QUESTION)", "(NEUTRAL)")


def _extract_between(text: str, start: str, end: str) -> str:
    if start not in text:
        raise ValueError(f"Could not find {start!r} in rating prompt.")
    tail = text.split(start, 1)[1]
    if end not in tail:
        raise ValueError(f"Could not find {end!r} after {start!r} in rating prompt.")
    return tail.split(end, 1)[0].strip()


def _parse_rating_prompt(prompt_text: str) -> dict:
    conversation_text = _extract_between(
        prompt_text,
        "Human-Chatbot conversation:\n",
        "\n\nThere are ",
    )
    conversation_lines = conversation_text.splitlines()
    if len(conversation_lines) != 7:
        raise ValueError(f"Expected 7 conversation lines, found {len(conversation_lines)}.")

    history = []
    for idx in range(0, 6, 2):
        human = conversation_lines[idx].removeprefix("Human: ").strip()
        chatbot = conversation_lines[idx + 1].removeprefix("Chatbot: ").strip()
        history.append([human, _restore_response_tags(chatbot, human)])
    prompt = conversation_lines[6].removeprefix("Human: ").strip()

    answers_text = _extract_between(
        prompt_text,
        "different possible chatbot answers, from A1 to A",
        "\n\nBase the scoring",
    )
    answers_text = "\n".join(answers_text.splitlines()[1:])
    matches = list(
        re.finditer(
            r"^A(?P<num>\d+)\s*[-:]\s*(?P<response>.*?)(?=^A\d+\s*[-:]|\Z)",
            answers_text,
            flags=re.MULTILINE | re.DOTALL,
        )
    )
    if not matches:
        raise ValueError("Could not parse candidate responses from rating prompt.")

    responses: dict[int, str] = {}
    for match in matches:
        idx = int(match.group("num"))
        responses[idx] = _restore_response_tags(match.group("response").strip(), prompt)

    return {
        "history": history,
        "prompt": prompt,
        "responses": responses,
    }


def _context_key(history: list, prompt: str) -> tuple:
    return tuple(tuple(row) for row in history), prompt


def _load_original_context(
    is_test: bool,
) -> tuple[dict[str, pd.Series], dict[tuple, pd.Series], dict[str, pd.Series]]:
    path = f"data/{with_suffix('rm_preference_dataset_original', 'json', is_test)}"
    original = pd.read_json(path)
    by_dialogue_id: dict[str, pd.Series] = {}
    by_context: dict[tuple, pd.Series] = {}
    by_prompt_candidates: dict[str, list[pd.Series]] = {}
    for _, row in original.iterrows():
        by_dialogue_id[str(row["dialogue_id"])] = row
        by_context[_context_key(row["history"], row["prompt"])] = row
        by_prompt_candidates.setdefault(row["prompt"], []).append(row)
    by_prompt = {
        prompt: rows[0]
        for prompt, rows in by_prompt_candidates.items()
        if len(rows) == 1
    }
    return by_dialogue_id, by_context, by_prompt


def _source_set(identity: str, original_row: pd.Series | None) -> str | None:
    if original_row is not None:
        return _series_value(original_row, "set", _series_value(original_row, "SET", None))
    return "sft-demonstration"


def combine_pairs_with_responses(is_test: bool) -> None:
    pairwise_df = pd.read_csv(f"data/{with_suffix('rm_preference_dataset_df', 'csv', is_test)}")
    id_col = _identity_column(pairwise_df)
    rank_df = pd.read_csv(f"data/{with_suffix('rm_preference_dataset_models_results_rank_modified', 'csv', is_test)}")
    rank_by_id = {str(row[id_col]): row for _, row in rank_df.iterrows()}
    missing_rank_ids = sorted(set(pairwise_df[id_col].astype(str)) - set(rank_by_id))
    if missing_rank_ids:
        raise ValueError(f"Pairwise IDs missing from rank data: {missing_rank_ids[:5]}")
    original_by_dialogue_id, original_by_context, original_by_prompt = _load_original_context(is_test)

    instructions, histories, prompts = [], [], []
    winner_responses, loser_responses = [], []
    dialogue_ids, source_sets = [], []
    context_validation_mismatches: list[str] = []

    for _, row in pairwise_df.iterrows():
        identity = str(row[id_col])
        parsed = _parse_rating_prompt(rank_by_id[identity]["PROMPT"])
        parsed_context_key = _context_key(parsed["history"], parsed["prompt"])
        original_row = original_by_dialogue_id.get(identity)
        if original_row is not None and _context_key(original_row["history"], original_row["prompt"]) != parsed_context_key:
            context_validation_mismatches.append(identity)
        if original_row is None:
            original_row = original_by_context.get(parsed_context_key)
        if original_row is None:
            original_row = original_by_prompt.get(parsed["prompt"])
        if original_row is not None:
            instructions.append(_series_value(original_row, "instruction", DEFAULT_SYSTEM))
            histories.append(original_row["history"])
            prompts.append(original_row["prompt"])
        else:
            instructions.append(DEFAULT_SYSTEM)
            histories.append(parsed["history"])
            prompts.append(parsed["prompt"])
        dialogue_ids.append(identity)
        source_sets.append(_source_set(identity, original_row))

        for response_list, idx in (
            (winner_responses, row["WINNER"]),
            (loser_responses, row["LOSER"]),
        ):
            response_idx = int(idx)
            if response_idx in parsed["responses"]:
                response_list.append(parsed["responses"][response_idx])
            elif response_idx == 10 and original_row is not None:
                response_list.append(original_row["predict_9"])
            else:
                raise KeyError(
                    f"Could not recover response A{response_idx} for dialogue {identity}."
                )

    pairwise_df["SYSTEM"] = instructions
    pairwise_df["HISTORY"] = histories
    pairwise_df["PROMPT"] = prompts
    pairwise_df["WINNER_RESPONSE"] = winner_responses
    pairwise_df["LOSER_RESPONSE"] = loser_responses
    pairwise_df["DIALOGUE_ID"] = dialogue_ids
    pairwise_df["set"] = source_sets
    pairwise_df = pairwise_df[
        pairwise_df["WINNER_RESPONSE"].astype(str)
        != pairwise_df["LOSER_RESPONSE"].astype(str)
    ].copy()

    pairwise_df.insert(0, "PREFERENCE_ID", pairwise_df.groupby(id_col).cumcount())
    pairwise_df["PREFERENCE_ID"] = (
        pairwise_df[id_col].map(_preference_id_prefix)
        + "-"
        + pairwise_df["PREFERENCE_ID"].astype(str).str.zfill(4)
    )

    pairwise_df.to_csv(f"data/{with_suffix('rm_preference_dataset_response', 'csv', is_test)}", index=False)
    pairwise_df = pairwise_df.rename(columns={"DIALOGUE_ID": "dialogue_id"})

    json_records = json.loads(pairwise_df.to_json(orient="records"))
    out_path = f"data/{with_suffix('rm_preference_dataset_response', 'json', is_test)}"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, ensure_ascii=False, indent=2)

    if context_validation_mismatches:
        sample = sorted(set(context_validation_mismatches))[:5]
        print(
            "WARNING: matched original rows by DIALOGUE_ID, but exact context "
            f"text differed for {len(set(context_validation_mismatches))} dialogue(s): {sample}"
        )


def postprocess(is_test: bool = False) -> None:
    add_modified_rank(is_test)
    build_pairwise_dataframe(is_test)
    combine_pairs_with_responses(is_test)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    postprocess(is_test=args.test)
