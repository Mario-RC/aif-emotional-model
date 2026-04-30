"""Post-process the per-row RANK into pairwise winner/loser comparisons + responses."""

from __future__ import annotations

import argparse
import ast
import json

import pandas as pd

from _lib import create_pairwise_comparisons, with_suffix


def _add_extra_value_for_modified(row: pd.Series) -> pd.Series:
    """For entries flagged as ``MODIFIED``, append a sentinel rank to mark the modified response."""
    if row["MODIFIED"]:
        rank_list = ast.literal_eval(row["RANK"])
        rank_list.append(max(rank_list) + 1)
        row["RANK"] = rank_list
    return row


def add_modified_rank(is_test: bool) -> pd.DataFrame:
    in_path = f"data/{with_suffix('rm_preference_dataset_models_results_rank', 'csv', is_test)}"
    out_path = f"data/{with_suffix('rm_preference_dataset_models_results_rank_modified', 'csv', is_test)}"

    df = pd.read_csv(in_path).apply(_add_extra_value_for_modified, axis=1)
    df.to_csv(out_path, index=False)
    return df


def build_pairwise_dataframe(is_test: bool) -> pd.DataFrame:
    in_path = f"data/{with_suffix('rm_preference_dataset_models_results_rank_modified', 'csv', is_test)}"
    df = pd.read_csv(in_path)

    rows: list[dict] = []
    for _, row in df.iterrows():
        ranking = ast.literal_eval(row["RANK"])
        for winner, loser in create_pairwise_comparisons(ranking):
            rows.append({"DID": row["DID"], "WINNER": winner, "LOSER": loser})
    pairwise_df = pd.DataFrame(rows)
    pairwise_df.to_csv(f"data/{with_suffix('rm_preference_dataset_df', 'csv', is_test)}", index=False)
    return pairwise_df


def combine_pairs_with_responses(is_test: bool) -> None:
    pairwise_df = pd.read_csv(f"data/{with_suffix('rm_preference_dataset_df', 'csv', is_test)}")
    original = pd.read_json(f"data/{with_suffix('rm_preference_dataset_original', 'json', is_test)}")

    instructions, histories, prompts = [], [], []
    winner_responses, loser_responses = [], []

    for _, row in pairwise_df.iterrows():
        did = row["DID"]
        original_row = original.loc[original["did"] == did]
        instructions.append(original_row["instruction"].values[0])
        histories.append(original_row["history"].values[0])
        prompts.append(original_row["prompt"].values[0])

        for response_list, idx in (
            (winner_responses, row["WINNER"]),
            (loser_responses, row["LOSER"]),
        ):
            key = "target" if idx == 1 else f"predict_{idx - 1}"
            response_list.append(original_row[key].values[0])

    pairwise_df["SYSTEM"] = instructions
    pairwise_df["HISTORY"] = histories
    pairwise_df["PROMPT"] = prompts
    pairwise_df["WINNER_RESPONSE"] = winner_responses
    pairwise_df["LOSER_RESPONSE"] = loser_responses

    # Replace DID prefix to a comparison-data-specific UID prefix.
    pairwise_df.insert(0, "UID", pairwise_df.groupby("DID").cumcount())
    pairwise_df["UID"] = (
        pairwise_df["DID"] + "-" + pairwise_df["UID"].astype(str).str.zfill(4)
    )
    pairwise_df["UID"] = pairwise_df["UID"].str.replace("GPT4", "COMPAR")
    pairwise_df["DID"] = pairwise_df["DID"].str.replace("GPT4", "COMPAR")

    pairwise_df.to_csv(f"data/{with_suffix('rm_preference_dataset_response', 'csv', is_test)}", index=False)
    pairwise_df = pairwise_df.drop(columns=["DID"])

    json_records = ast.literal_eval(pairwise_df.to_json(orient="records"))
    out_path = f"data/{with_suffix('rm_preference_dataset_response', 'json', is_test)}"
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
