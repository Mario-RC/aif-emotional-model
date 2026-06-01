"""Post-process the per-row RANK into pairwise winner/loser comparisons + responses."""

from __future__ import annotations

import argparse
import ast
import json

import pandas as pd

from _lib import create_pairwise_comparisons, with_suffix, write_csv

IDENTITY_COLUMN = "DIALOGUE_ID"


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


def _lookup_original_row(original: pd.DataFrame, identity: str) -> pd.Series:
    rows = original.loc[original["dialogue_id"].astype(str) == str(identity)]
    if not rows.empty:
        return rows.iloc[0]
    raise KeyError(f"Could not find original row for dialogue identity {identity!r}.")


def _preference_id_prefix(identity: str) -> str:
    text = str(identity)
    if "-" in text:
        _, suffix = text.split("-", 1)
        return f"RLAIFEP-{suffix}"
    return f"RLAIFEP-{text}"


def _validate_dialogue_ids(df: pd.DataFrame, original: pd.DataFrame, id_col: str) -> None:
    """Fail if rank/pairwise IDs do not exactly match the source dataset."""
    source_ids = set(original["dialogue_id"].astype(str))
    data_ids = set(df[id_col].astype(str))
    missing = sorted(source_ids - data_ids)
    extra = sorted(data_ids - source_ids)
    if missing or extra:
        raise ValueError(
            "Rank outputs do not match dpo_preference_dataset_original IDs. "
            f"missing={missing[:5]}, extra={extra[:5]}. "
            "Do not remap legacy IDs silently; regenerate rating CSVs from the "
            "current source dataset."
        )


def add_modified_rank(is_test: bool) -> pd.DataFrame:
    in_path = f"data/{with_suffix('dpo_preference_dataset_models_results_rank', 'csv', is_test)}"
    out_path = f"data/{with_suffix('dpo_preference_dataset_models_results_rank_modified', 'csv', is_test)}"

    df = pd.read_csv(in_path)
    original = pd.read_json(f"data/{with_suffix('dpo_preference_dataset_original', 'json', is_test)}")
    _validate_dialogue_ids(df, original, _identity_column(df))
    df = df.apply(_add_extra_value_for_modified, axis=1)
    write_csv(df, out_path)
    return df


def build_pairwise_dataframe(is_test: bool) -> pd.DataFrame:
    in_path = f"data/{with_suffix('dpo_preference_dataset_models_results_rank_modified', 'csv', is_test)}"
    df = pd.read_csv(in_path)
    id_col = _identity_column(df)

    rows: list[dict] = []
    for _, row in df.iterrows():
        ranking = ast.literal_eval(row["RANK"])
        for winner, loser in create_pairwise_comparisons(ranking):
            rows.append({IDENTITY_COLUMN: row[id_col], "WINNER": winner, "LOSER": loser})
    pairwise_df = pd.DataFrame(rows)
    write_csv(pairwise_df, f"data/{with_suffix('dpo_preference_dataset_df', 'csv', is_test)}")
    return pairwise_df


def combine_pairs_with_responses(is_test: bool) -> None:
    pairwise_df = pd.read_csv(f"data/{with_suffix('dpo_preference_dataset_df', 'csv', is_test)}")
    original = pd.read_json(f"data/{with_suffix('dpo_preference_dataset_original', 'json', is_test)}")
    id_col = _identity_column(pairwise_df)
    _validate_dialogue_ids(pairwise_df, original, id_col)

    instructions, histories, prompts = [], [], []
    winner_responses, loser_responses = [], []
    dialogue_ids, source_sets = [], []

    for _, row in pairwise_df.iterrows():
        identity = row[id_col]
        original_row = _lookup_original_row(original, identity)
        instructions.append(original_row["instruction"])
        histories.append(original_row["history"])
        prompts.append(original_row["prompt"])
        dialogue_ids.append(_series_value(original_row, "dialogue_id", str(identity)))
        source_sets.append(_series_value(original_row, "set", _series_value(original_row, "SET", None)))

        for response_list, idx in (
            (winner_responses, row["WINNER"]),
            (loser_responses, row["LOSER"]),
        ):
            key = "target" if idx == 1 else f"predict_{idx - 1}"
            response_list.append(original_row[key])

    pairwise_df["SYSTEM"] = instructions
    pairwise_df["HISTORY"] = histories
    pairwise_df["PROMPT"] = prompts
    pairwise_df["WINNER_RESPONSE"] = winner_responses
    pairwise_df["LOSER_RESPONSE"] = loser_responses
    pairwise_df["DIALOGUE_ID"] = dialogue_ids
    pairwise_df["set"] = source_sets

    pairwise_df.insert(0, "PREFERENCE_ID", pairwise_df.groupby(id_col).cumcount())
    pairwise_df["PREFERENCE_ID"] = (
        pairwise_df[id_col].map(_preference_id_prefix)
        + "-"
        + pairwise_df["PREFERENCE_ID"].astype(str).str.zfill(4)
    )

    write_csv(pairwise_df, f"data/{with_suffix('dpo_preference_dataset_response', 'csv', is_test)}")
    pairwise_df = pairwise_df.rename(columns={"DIALOGUE_ID": "dialogue_id"})

    json_records = json.loads(pairwise_df.to_json(orient="records"))
    out_path = f"data/{with_suffix('dpo_preference_dataset_response', 'json', is_test)}"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, ensure_ascii=False, indent=2)


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
