"""Convert per-LLM rating CSVs into the per-row aggregated rank.

Pipeline:
1. ``parse_llm_rates``: parse the textual ``COMPLETION`` column into
   ``RATE`` (raw scores) and ``TRANSFORMED_RATE`` (expression-table scaled).
2. ``combine_models``: stack the per-LLM rate columns into a single dataframe.
3. ``normalize_and_rescale``: z-score each LLM's RATEs and re-shift to 1-9.
4. ``compute_overall_rank``: sum across criteria, average across LLMs, rank.
5. ``plot_rates``: optional violin / heatmap plots.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata

from _lib import expression_table, parse_completion_rates, with_suffix

LLM_NAMES = ["GPT-4O", "CLAUDE-3.5-SONNET", "GEMINI-1.5-PRO", "LLAMA-3.1-405B"]


def _row_identity(row: pd.Series) -> str:
    return str(row.get("DIALOGUE_ID", "<unknown>"))


# ---------------------------------------------------------------------------
# Stage 1: parse the per-LLM rating CSV into RATE / TRANSFORMED_RATE columns.
# ---------------------------------------------------------------------------

def _rating_csv_path(llm_name: str, is_test: bool) -> Path:
    path = Path("data") / llm_name / with_suffix(
        f"rm_preference_dataset_rate_{llm_name}", "csv", is_test
    )
    if path.exists():
        return path

    raise FileNotFoundError(
        f"Missing rating CSV for {llm_name}: {path}. "
        "Use the paid rating CSVs renamed to rm_preference_dataset_rate_*."
    )


def _validate_rating_alignment(per_llm: dict[str, pd.DataFrame]) -> None:
    """Ensure paid rating CSVs describe the same rows in the same order."""
    base_llm = LLM_NAMES[0]
    base = per_llm[base_llm]
    base_ids = base["DIALOGUE_ID"].astype(str).tolist()
    if len(base_ids) != len(set(base_ids)):
        duplicated = base.loc[base["DIALOGUE_ID"].astype(str).duplicated(), "DIALOGUE_ID"]
        raise ValueError(f"{base_llm} rating CSV has duplicated DIALOGUE_ID values: {duplicated.head().tolist()}")

    base_prompts = base["PROMPT"].astype(str).tolist()
    for llm in LLM_NAMES[1:]:
        df = per_llm[llm]
        ids = df["DIALOGUE_ID"].astype(str).tolist()
        prompts = df["PROMPT"].astype(str).tolist()
        if ids != base_ids:
            raise ValueError(f"{llm} rating CSV DIALOGUE_ID order does not match {base_llm}.")
        if prompts != base_prompts:
            raise ValueError(f"{llm} rating CSV PROMPT order does not match {base_llm}.")


def parse_llm_rates(llm_name: str, is_test: bool) -> pd.DataFrame:
    """Read a per-LLM rating CSV and add RATE / TRANSFORMED_RATE columns."""
    csv_path = _rating_csv_path(llm_name, is_test)
    df = pd.read_csv(csv_path)
    df.rename(columns={"RANK": "COMPLETION"}, inplace=True)

    # Llama-3.1-405B occasionally adds stray "-" tokens that must be stripped.
    extra_clean = ("-",) if llm_name == "LLAMA-3.1-405B" else ()

    rates_list, transformed_rates_list = [], []
    for raw_completion, expression_levels in zip(df["COMPLETION"], df["EXPRESSION_LEVEL"]):
        raw_text = ast.literal_eval(raw_completion)[0]
        expression_levels = ast.literal_eval(expression_levels)
        rate_list = parse_completion_rates(raw_text, extra_clean=extra_clean)
        transformed = [expression_table(rate, expression_levels) for rate in rate_list]
        rates_list.append(rate_list)
        transformed_rates_list.append(transformed)

    df.insert(6, "RATE", rates_list)
    df.insert(7, "TRANSFORMED_RATE", transformed_rates_list)
    return df


# ---------------------------------------------------------------------------
# Stage 2: combine all LLMs' RATE / TRANSFORMED_RATE columns into one frame.
# ---------------------------------------------------------------------------

def combine_models(per_llm: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = per_llm[LLM_NAMES[0]].copy()
    base.drop(columns=["COMPLETION"], inplace=True)
    base.rename(
        columns={
            "RATE": f"RATE_{LLM_NAMES[0]}",
            "TRANSFORMED_RATE": f"RATE_{LLM_NAMES[0]}_TRANSFORMED",
        },
        inplace=True,
    )

    for offset, llm in enumerate(LLM_NAMES[1:], start=1):
        rate_col = 7 + (offset - 1) * 2
        base.insert(rate_col, f"RATE_{llm}", per_llm[llm]["RATE"])
        base.insert(rate_col + 1, f"RATE_{llm}_TRANSFORMED", per_llm[llm]["TRANSFORMED_RATE"])
    return base


# ---------------------------------------------------------------------------
# Stage 3: z-score per LLM, then rescale back to a 1-9 integer scale.
# ---------------------------------------------------------------------------

def _zscore_per_criterion(rates_column: pd.Series) -> list[list[list[float]]]:
    total_e, total_em, total_q = [], [], []
    for rates in rates_column:
        rates = ast.literal_eval(rates) if isinstance(rates, str) else rates
        total_e += [r[0] for r in rates]
        total_em += [r[1] for r in rates]
        total_q += [r[2] for r in rates]

    m_e, s_e = mean(total_e), stdev(total_e)
    m_em, s_em = mean(total_em), stdev(total_em)
    m_q, s_q = mean(total_q), stdev(total_q)

    out: list[list[list[float]]] = []
    for rates in rates_column:
        rates = ast.literal_eval(rates) if isinstance(rates, str) else rates
        out.append(
            [
                [
                    (r[0] - m_e) / s_e,
                    (r[1] - m_em) / s_em,
                    (r[2] - m_q) / s_q,
                ]
                for r in rates
            ]
        )
    return out


def _rescale_to_1_9(rates: list[list[list[float]]]) -> list[list[list[int]]]:
    return [[[int(np.round(v + 5)) for v in r] for r in row] for row in rates]


def normalize_and_rescale(df: pd.DataFrame) -> pd.DataFrame:
    """For each LLM, add ``RATE_<LLM>_NORMALIZED`` and ``RATE_<LLM>_RESCALED`` columns."""
    insert_at = 7
    for llm in LLM_NAMES:
        normalized = _zscore_per_criterion(df[f"RATE_{llm}"])
        df.insert(insert_at, f"RATE_{llm}_NORMALIZED", normalized)
        rescaled = _rescale_to_1_9(normalized)
        df.insert(insert_at + 1, f"RATE_{llm}_RESCALED", rescaled)
        insert_at += 4
    return df


# ---------------------------------------------------------------------------
# Stage 4: aggregate across criteria and across LLMs to produce a final RANK.
# ---------------------------------------------------------------------------

def compute_overall_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``RATE_<LLM>_OVERALL`` (sum of three criteria) and final ``RANK`` columns."""
    overall_per_llm = {llm: [] for llm in LLM_NAMES}

    for _, row in df.iterrows():
        for llm in LLM_NAMES:
            rates = ast.literal_eval(row[f"RATE_{llm}_RESCALED"]) if isinstance(
                row[f"RATE_{llm}_RESCALED"], str
            ) else row[f"RATE_{llm}_RESCALED"]
            overall_per_llm[llm].append([sum(r) for r in rates])

    for offset, llm in enumerate(LLM_NAMES):
        df.insert(9 + offset * 5, f"RATE_{llm}_OVERALL", overall_per_llm[llm])

    rank_means, rank_stds, ranked = [], [], []
    for _, row in df.iterrows():
        n_responses_per_llm = {
            len(row[f"RATE_{llm}_OVERALL"])
            for llm in LLM_NAMES
        }
        if len(n_responses_per_llm) != 1:
            raise ValueError(
                f"Different number of scored responses for dialogue {_row_identity(row)}: "
                f"{sorted(n_responses_per_llm)}"
            )

        n_responses = n_responses_per_llm.pop()
        per_response = []
        for idx in range(n_responses):
            per_response.append([row[f"RATE_{llm}_OVERALL"][idx] for llm in LLM_NAMES])
        means_row = [mean(r) for r in per_response]
        stds_row = [stdev(r) for r in per_response]
        rank_means.append(means_row)
        rank_stds.append(stds_row)
        ranked.append(list(rankdata([-r for r in means_row], method="dense")))

    df["RATE_MEAN_OVERALL"] = rank_means
    df["RATE_STD_OVERALL"] = rank_stds
    df["RANK"] = ranked
    return df


# ---------------------------------------------------------------------------
# Stage 5: optional plots (used for inspection only).
# ---------------------------------------------------------------------------

def plot_rates(df: pd.DataFrame, column_suffix: str = "_RESCALED") -> None:
    flattened = {
        llm: np.concatenate(np.array(df[f"RATE_{llm}{column_suffix}"].apply(ast.literal_eval)))
        for llm in LLM_NAMES
    }

    plot_data, labels = [], []
    for llm, values in flattened.items():
        plot_data.extend(values)
        labels.extend([llm] * len(values))
    long_df = pd.DataFrame({"Score": plot_data, "Model": labels})

    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Model", y="Score", data=long_df.explode("Score"), inner="quart", palette="Set2")
    plt.title("Scores Across Models")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y="Score", data=long_df.explode("Score"), palette="Set2")
    plt.title("Scores Across Models")
    plt.show()

    means = [np.mean(values) for values in flattened.values()]
    std_devs = [np.std(values) for values in flattened.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(LLM_NAMES, means, yerr=std_devs, capsize=5, color="skyblue", edgecolor="black")
    plt.title("Mean Scores with Standard Deviation Across Models")
    plt.show()

    differences = np.zeros((len(LLM_NAMES), len(LLM_NAMES)))
    for i, m_i in enumerate(LLM_NAMES):
        for j, m_j in enumerate(LLM_NAMES):
            if i != j:
                differences[i, j] = np.mean(flattened[m_i] - flattened[m_j])
    plt.figure(figsize=(8, 6))
    sns.heatmap(differences, annot=True, xticklabels=LLM_NAMES, yticklabels=LLM_NAMES, cmap="coolwarm", center=0)
    plt.show()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def rates_to_ranks(is_test: bool = False, with_plots: bool = False) -> None:
    per_llm = {llm: parse_llm_rates(llm, is_test) for llm in LLM_NAMES}
    _validate_rating_alignment(per_llm)
    df = combine_models(per_llm)
    df.to_csv(f"data/{with_suffix('rm_preference_dataset_models_results', 'csv', is_test)}", index=False)

    df = pd.read_csv(f"data/{with_suffix('rm_preference_dataset_models_results', 'csv', is_test)}")
    df = normalize_and_rescale(df)
    df.to_csv(f"data/{with_suffix('rm_preference_dataset_models_results_nsr', 'csv', is_test)}", index=False)

    df = pd.read_csv(f"data/{with_suffix('rm_preference_dataset_models_results_nsr', 'csv', is_test)}")
    df = compute_overall_rank(df)
    df.to_csv(f"data/{with_suffix('rm_preference_dataset_models_results_rank', 'csv', is_test)}", index=False)

    if with_plots:
        plot_rates(df)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rates_to_ranks(is_test=args.test, with_plots=args.plots)
