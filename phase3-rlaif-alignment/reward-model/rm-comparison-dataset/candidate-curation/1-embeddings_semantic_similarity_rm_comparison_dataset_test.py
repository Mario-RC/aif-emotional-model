"""Test-split entry point for :mod:`1-embeddings_semantic_similarity_rm_comparison_dataset`."""

from __future__ import annotations

import importlib

embeddings_module = importlib.import_module("1-embeddings_semantic_similarity_rm_comparison_dataset")


if __name__ == "__main__":
    embeddings_module.run_pipeline(is_test=True)
