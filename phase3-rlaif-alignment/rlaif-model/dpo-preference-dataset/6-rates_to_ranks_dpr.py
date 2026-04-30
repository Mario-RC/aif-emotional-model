"""Plot rates across all four data splits combined (train + test, comparison + rlaif).

Reads the four ``*_models_results_rank[*test].csv`` produced by
:mod:`3-rates_to_ranks` and renders violin / box / bar / scatter / heatmap
plots for both the per-LLM ``RESCALED`` rates and the per-LLM ``OVERALL``
aggregated scores.
"""

from __future__ import annotations

import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LLM_NAMES = ["GPT-4O", "CLAUDE-3.5-SONNET", "GEMINI-1.5-PRO", "LLAMA-3.1-405B"]
LLM_LABELS = ["GPT-4o", "Claude 3.5 Sonnet", "Gemini 1.5 Pro", "Llama 3.1 405B"]


def load_combined() -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_csv("data/comparison_data_models_results_rank.csv"),
            pd.read_csv("data/comparison_data_models_results_rank_test.csv"),
            pd.read_csv("data/dpo_preference_dataset_models_results_rank.csv"),
            pd.read_csv("data/dpo_preference_dataset_models_results_rank_test.csv"),
        ],
        ignore_index=True,
    )


def _flatten_per_model(df: pd.DataFrame, suffix: str) -> dict[str, np.ndarray]:
    data = {
        label: np.array(df[f"RATE_{key}{suffix}"].apply(ast.literal_eval))
        for label, key in zip(LLM_LABELS, LLM_NAMES)
    }
    return {label: np.concatenate(arrays) for label, arrays in data.items()}


def _plot_violins_and_box(flattened: dict[str, np.ndarray], explode: bool) -> None:
    plot_data, labels = [], []
    for model, values in flattened.items():
        plot_data.extend(values)
        labels.extend([model] * len(values))
    long_df = pd.DataFrame({"Score": plot_data, "Model": labels})
    if explode:
        long_df = long_df.explode("Score")

    for inner in ("quart", "box"):
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="Model", y="Score", data=long_df, inner=inner, palette="Set2")
        plt.title("Scores Across Models")
        plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y="Score", data=long_df, palette="Set2")
    plt.title("")
    plt.show()


def _plot_bar_and_heatmap(flattened: dict[str, np.ndarray]) -> None:
    means = [np.mean(values) for values in flattened.values()]
    std_devs = [np.std(values) for values in flattened.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(list(flattened.keys()), means, yerr=std_devs, capsize=5, color="skyblue", edgecolor="black")
    plt.title("Mean Scores with Standard Deviation Across Models")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(flattened["GPT-4o"], flattened["Claude 3.5 Sonnet"], alpha=0.5)
    plt.title("Score Comparison: GPT-4o vs. Claude 3.5 Sonnet")
    plt.plot([1, 9], [1, 9], color="red", linestyle="--", label="Perfect Alignment")
    plt.legend()
    plt.show()

    labels = list(flattened.keys())
    differences = np.zeros((len(labels), len(labels)))
    for i, m_i in enumerate(labels):
        for j, m_j in enumerate(labels):
            if i != j:
                differences[i, j] = np.mean(flattened[m_i] - flattened[m_j])
    plt.figure(figsize=(8, 6))
    sns.heatmap(differences, annot=True, xticklabels=labels, yticklabels=labels, cmap="coolwarm", center=0)
    plt.title("")
    plt.show()


def plot_all() -> None:
    df = load_combined()

    # Rescaled per-criterion plots (need to explode list-typed scores).
    rescaled = _flatten_per_model(df, suffix="_RESCALED")
    _plot_violins_and_box(rescaled, explode=True)
    _plot_bar_and_heatmap(rescaled)

    # Overall (sum across criteria) plots.
    overall = _flatten_per_model(df, suffix="_OVERALL")
    _plot_violins_and_box(overall, explode=False)
    _plot_bar_and_heatmap(overall)


if __name__ == "__main__":
    plot_all()
