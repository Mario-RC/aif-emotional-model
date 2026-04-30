"""Test-split entry point for :mod:`4-postprocess_rm_preference_dataset`."""

from __future__ import annotations

import importlib

postprocess_module = importlib.import_module("4-postprocess_rm_preference_dataset")


if __name__ == "__main__":
    postprocess_module.postprocess(is_test=True)
