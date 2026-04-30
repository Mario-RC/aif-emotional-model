"""Test-split entry point for :mod:`1-preprocess_rm_preference_dataset`."""

from __future__ import annotations

import importlib

preprocess_module = importlib.import_module("1-preprocess_rm_preference_dataset")


if __name__ == "__main__":
    preprocess_module.preprocess(is_test=True)
