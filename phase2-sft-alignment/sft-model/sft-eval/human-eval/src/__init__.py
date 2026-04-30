"""Reusable building blocks for the human-eval pipeline.

The original flow lived in two Jupyter notebooks with heavily copy-pasted
cells per annotator / per model / per task. This package extracts the
logic into small, testable modules so both the *generation* side
(build annotation spreadsheets) and the *analysis* side (aggregate the
filled-in spreadsheets into metrics and plots) share the same primitives.
"""
