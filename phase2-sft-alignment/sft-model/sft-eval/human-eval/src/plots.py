"""Matplotlib helpers for the human-eval metrics.

Two chart families:

* ``plot_hits_by_model`` — grouped bar chart with one bar per
  (human, model1, ..., modelN), optionally with stdev error bars and
  saved to PDF.
* ``plot_hits_by_emotion`` — one bar per emotion label, coloured via
  ``EMOTION_COLORS``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .config import EMOTION_COLORS, HIST_DIR, MODEL_LABELS


DEFAULT_BAR_COLOR = "#b6e3d1"
DEFAULT_YTICKS = [0, 5, 15, 25, 35, 45]


def _finalize_axes() -> None:
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_hits_by_model(
    hits: Sequence[float],
    title: str = "",
    ylim: float = 45,
    stds: Optional[Sequence[float]] = None,
    bar_color: str = DEFAULT_BAR_COLOR,
    error_color: str = "orange",
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (8, 3.5),
    columns: Sequence[str] = MODEL_LABELS,
) -> None:
    """Bar chart of hit counts across [human, model1, model2, ...]."""
    plt.figure(figsize=figsize)
    plt.ylim([0, ylim])
    plt.title(title, fontsize=16)
    plt.xlabel("Models", fontsize=12)
    plt.ylabel("Hit", fontsize=12)
    x = np.arange(len(hits))
    if stds is not None:
        plt.bar(x, height=hits, yerr=stds, color=bar_color,
                capsize=5, ecolor=error_color)
    else:
        plt.bar(x, height=hits, color=bar_color)
    plt.xticks(x, list(columns), fontsize=12)
    plt.yticks(DEFAULT_YTICKS, fontsize=12)
    _finalize_axes()
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.show()
    plt.close()


def plot_hits_by_emotion(
    hits_by_emotion: Dict[str, float],
    title: str = "",
    ylim: float = 45,
    stds_by_emotion: Optional[Dict[str, float]] = None,
    colors: Dict[str, str] = EMOTION_COLORS,
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (8, 3.5),
    error_color: str = "gray",
) -> None:
    """Bar chart with one coloured bar per emotion."""
    labels = list(hits_by_emotion.keys())
    values = list(hits_by_emotion.values())
    bar_colors = [colors.get(label, "black") for label in labels]

    plt.figure(figsize=figsize)
    plt.ylim([0, ylim])
    plt.title(title, fontsize=16)
    plt.xlabel("Emotion", fontsize=12)
    plt.ylabel("Hit", fontsize=12)
    if stds_by_emotion is not None:
        err = [stds_by_emotion.get(e, 0) for e in labels]
        plt.bar(labels, values, color=bar_colors, yerr=err,
                capsize=5, ecolor=error_color)
    else:
        plt.bar(labels, values, color=bar_colors)
    plt.xticks(fontsize=12)
    plt.yticks(DEFAULT_YTICKS, fontsize=12)
    _finalize_axes()
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    plt.show()
    plt.close()


# Convenience wrappers around ``HIST_DIR``.
def save_path_models() -> Path:
    return HIST_DIR / "models" / "hist_models.pdf"


def save_path_emotions() -> Path:
    return HIST_DIR / "emotions" / "hist_emotions.pdf"
