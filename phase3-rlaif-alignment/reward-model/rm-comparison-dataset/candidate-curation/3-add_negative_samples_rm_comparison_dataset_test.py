"""Test-split entry point for :mod:`3-add_negative_samples_rm_comparison_dataset`."""

from __future__ import annotations

import importlib

negative_module = importlib.import_module("3-add_negative_samples_rm_comparison_dataset")


if __name__ == "__main__":
    args = negative_module._parse_args()
    negative_module.add_negative_samples(
        is_test=True,
        llama_model=args.llama_model,
        device=args.device,
        skip_llm=args.skip_llm,
        llm_batch_size=args.llm_batch_size,
        checkpoint_every=args.checkpoint_every,
    )
