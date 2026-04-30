"""Test-split entry point for :mod:`2-generate_rating_data`."""

from __future__ import annotations

import importlib

generate_module = importlib.import_module("2-generate_rating_data")


if __name__ == "__main__":
    args = generate_module._parse_args()
    args.test = True
    generate_module.generate_rating_data(
        llm_name=args.llm,
        is_test=True,
        config_file=args.config,
        start_offset=args.start,
    )
