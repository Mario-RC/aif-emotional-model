"""Test-split entry point for :mod:`2-delete_respones_dpo_comparison_dataset`."""

from __future__ import annotations

import importlib

filter_module = importlib.import_module("2-delete_respones_dpo_comparison_dataset")


if __name__ == "__main__":
    filter_module.run_pipeline(is_test=True)
