"""Test-split entry point for :mod:`5-format_dpo_preference_dataset`."""

from __future__ import annotations

import importlib

format_module = importlib.import_module("5-format_dpo_preference_dataset")


if __name__ == "__main__":
    format_module.format_dpo_preference_dataset(is_test=True)
