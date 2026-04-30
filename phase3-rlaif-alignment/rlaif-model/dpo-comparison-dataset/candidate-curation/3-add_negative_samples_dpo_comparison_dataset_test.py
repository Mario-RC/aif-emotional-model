"""Test-split entry point for :mod:`3-add_negative_samples_dpo_comparison_dataset`."""

from __future__ import annotations

import importlib

negative_module = importlib.import_module("3-add_negative_samples_dpo_comparison_dataset")


if __name__ == "__main__":
    negative_module.add_negative_samples(is_test=True)
