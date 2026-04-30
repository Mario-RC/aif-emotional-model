"""Test-split entry point for :mod:`3-rates_to_ranks`."""

from __future__ import annotations

import importlib

rates_module = importlib.import_module("3-rates_to_ranks")


if __name__ == "__main__":
    args = rates_module._parse_args()
    rates_module.rates_to_ranks(is_test=True, with_plots=args.plots)
