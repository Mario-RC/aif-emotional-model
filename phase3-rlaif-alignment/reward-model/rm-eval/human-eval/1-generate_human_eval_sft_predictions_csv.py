"""Regenerate the wide SFT-predictions CSV used by RM human evaluation.

This script builds ``data/rm_human_eval_sft_predictions_test.csv``
from:

* Phase 3 prompt JSON files with ``instruction`` / ``output`` rows. By
  default these are read from the canonical PPO prompt dataset under
  ``phase3-rlaif-alignment/rlaif-model/ppo-unlabeled-prompts-dataset/data``.
* One SFT prediction JSON/JSONL per model.
* Optionally, a lightweight selection CSV with ``SEG`` / ``TARGET`` columns
  to preserve the exact prompt subset and order used for annotation.

The prediction files are expected to contain one record per prompt with these
fields, or close equivalents:

* prompt: ``input`` or ``instruction``
* reference answer: ``target``, ``output`` or ``label``
* SFT answer: ``predict_sft``, ``predict`` or ``prediction``

Example:

    python 1-generate_human_eval_sft_predictions_csv.py \\
      --model-result GLM4=results/sft_predictions/glm4.json \\
      --model-result GEMMA=results/sft_predictions/gemma.json \\
      --model-result LLAMA3=results/sft_predictions/llama3.json \\
      --model-result MISTRAL=results/sft_predictions/mistral.json \\
      --model-result PHI3=results/sft_predictions/phi3.json \\
      --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MODEL_ORDER = ("GLM4", "GEMMA", "LLAMA3", "MISTRAL", "PHI3")
DEFAULT_OUTPUT_NAME = "rm_human_eval_sft_predictions_test.csv"
EMOTIONS = ("ANGER", "DISGUST", "FEAR", "HAPPINESS", "NEUTRAL", "SADNESS", "SURPRISE")
EMOTION_TAG_RE = re.compile(r"\((" + "|".join(EMOTIONS) + r")\)", re.IGNORECASE)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
DATA_DIR = SCRIPT_DIR / "data"
PPO_PROMPTS_DATA_DIR = (
    PROJECT_ROOT
    / "phase3-rlaif-alignment"
    / "rlaif-model"
    / "ppo-unlabeled-prompts-dataset"
    / "data"
)
DEFAULT_PROMPT_JSONS = (
    PPO_PROMPTS_DATA_DIR / "ppo_unlabeled_prompts_dataset.json",
    PPO_PROMPTS_DATA_DIR / "ppo_unlabeled_prompts_dataset_test.json",
)
DEFAULT_OUTPUT_CSV = DATA_DIR / DEFAULT_OUTPUT_NAME


@dataclass(frozen=True)
class PromptRow:
    instruction: str
    target: str


def parse_single_tagged(text: str) -> tuple[str, str]:
    tag_match = EMOTION_TAG_RE.search(text)
    if not tag_match:
        raise ValueError(f"No emotion tag found in: {text!r}")
    return tag_match.group(1), text[tag_match.end():].strip()


def split_tagged_sentences(text: str, expected: int = 3) -> list[tuple[str, str]]:
    matches = list(EMOTION_TAG_RE.finditer(text))
    if len(matches) != expected:
        raise ValueError(
            f"Expected {expected} emotion tags, got {len(matches)} in: {text!r}"
        )

    pairs: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        pairs.append((match.group(1), text[start:end].strip()))
    return pairs


def load_json_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def pick(record: dict, keys: Iterable[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def load_prompt_rows(paths: Iterable[Path]) -> list[PromptRow]:
    rows: list[PromptRow] = []
    for path in paths:
        for record in load_json_records(path):
            instruction = pick(record, ("instruction", "input"))
            target = pick(record, ("output", "target", "label"))
            if instruction is None or target is None:
                raise ValueError(
                    f"Prompt row in {path} must contain instruction/input "
                    "and output/target/label"
                )
            rows.append(PromptRow(instruction=instruction, target=target))
    return rows


def apply_selection(prompt_rows: list[PromptRow], selection_csv: Path | None) -> list[PromptRow]:
    if selection_csv is None:
        return prompt_rows

    prompt_map = {
        (row.instruction, row.target): row
        for row in prompt_rows
    }
    selected: list[PromptRow] = []
    missing: list[tuple[str, str]] = []

    with selection_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "SEG" not in reader.fieldnames or "TARGET" not in reader.fieldnames:
            raise ValueError(f"{selection_csv} must contain SEG and TARGET columns")

        for row in reader:
            key = (row["SEG"], row["TARGET"])
            prompt = prompt_map.get(key)
            if prompt is None:
                missing.append(key)
            else:
                selected.append(prompt)

    if missing:
        raise ValueError(
            f"{selection_csv} contains {len(missing)} prompts that are not in "
            "the Phase 3 prompt JSON files"
        )
    return selected


def parse_model_result_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            "Use SHORT=path, for example GEMMA=results/sft_predictions/gemma.json"
        )
    short, path = raw.split("=", 1)
    short = short.strip().upper()
    if short not in MODEL_ORDER:
        raise argparse.ArgumentTypeError(
            f"Unknown model short name {short!r}. Expected one of: {', '.join(MODEL_ORDER)}"
        )
    return short, Path(path)


def load_model_predictions(path: Path) -> dict[tuple[str | None, str], str]:
    predictions: dict[tuple[str | None, str], str] = {}
    for record in load_json_records(path):
        prompt = pick(record, ("input", "instruction"))
        target = pick(record, ("target", "output", "label"))
        prediction = pick(record, ("predict_sft", "predict", "prediction", "response"))
        if target is None or prediction is None:
            raise ValueError(
                f"Prediction row in {path} must contain target/output/label "
                "and predict_sft/predict/prediction/response"
            )

        key = (prompt, target)
        if key in predictions and predictions[key] != prediction:
            raise ValueError(f"Duplicate prediction for the same prompt/target in {path}")
        predictions[key] = prediction
    return predictions


def find_prediction(
    predictions: dict[tuple[str | None, str], str],
    prompt: PromptRow,
) -> str | None:
    exact = predictions.get((prompt.instruction, prompt.target))
    if exact is not None:
        return exact

    # Some LLaMA-Factory outputs keep only the reference label and generated
    # text. This fallback is valid only when the target label is unique.
    target_matches = [
        prediction
        for (instruction, target), prediction in predictions.items()
        if instruction is None and target == prompt.target
    ]
    if len(target_matches) == 1:
        return target_matches[0]
    return None


def build_row(uid: str, prompt: PromptRow, model_predictions: dict[str, str]) -> dict[str, str]:
    human_emo, human_utt = parse_single_tagged(prompt.instruction)
    target_pairs = split_tagged_sentences(prompt.target)

    row = {
        "UID": uid,
        "SEG": prompt.instruction,
        "TARGET": prompt.target,
    }

    for short in MODEL_ORDER:
        row[f"SFT_PREDICTION_{short}"] = model_predictions[short]

    row["HUMAN_UTT"] = human_utt
    row["HUMAN_EMO"] = human_emo

    for idx, (emotion, utterance) in enumerate(target_pairs, start=1):
        row[f"TARGET_R{idx}"] = utterance
        row[f"TARGET_EMO{idx}"] = emotion

    parsed_predictions = {
        short: split_tagged_sentences(prediction)
        for short, prediction in model_predictions.items()
    }
    for idx in range(1, 4):
        for short in MODEL_ORDER:
            emotion, utterance = parsed_predictions[short][idx - 1]
            row[f"SFT_R{idx}_{short}"] = utterance
            row[f"SFT_EMO{idx}_{short}"] = emotion

    return row


def output_columns() -> list[str]:
    columns = [
        "UID", "SEG", "TARGET",
        *[f"SFT_PREDICTION_{short}" for short in MODEL_ORDER],
        "HUMAN_UTT", "HUMAN_EMO",
        "TARGET_R1", "TARGET_EMO1",
        "TARGET_R2", "TARGET_EMO2",
        "TARGET_R3", "TARGET_EMO3",
    ]
    for idx in range(1, 4):
        for short in MODEL_ORDER:
            columns.extend([f"SFT_R{idx}_{short}", f"SFT_EMO{idx}_{short}"])
    return columns


def build_wide_rows(
    prompt_rows: list[PromptRow],
    predictions_by_model: dict[str, dict[tuple[str | None, str], str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    missing: dict[str, int] = {short: 0 for short in MODEL_ORDER}

    for prompt in prompt_rows:
        model_predictions: dict[str, str] = {}
        for short in MODEL_ORDER:
            prediction = find_prediction(predictions_by_model[short], prompt)
            if prediction is None:
                missing[short] += 1
            else:
                model_predictions[short] = prediction

        if len(model_predictions) == len(MODEL_ORDER):
            uid = f"SFTANNO-{len(rows):06d}"
            rows.append(build_row(uid, prompt, model_predictions))

    if not rows:
        raise ValueError(
            "No rows could be built. Check that the prompt JSONs and SFT "
            "prediction files use the same instruction/target text."
        )

    if any(missing.values()):
        missing_msg = ", ".join(f"{short}: {count}" for short, count in missing.items())
        print(f"Skipped prompts without complete model predictions ({missing_msg})")
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path, overwrite: bool) -> Path:
    final_path = output_path
    if output_path.exists() and not overwrite:
        final_path = output_path.with_name(
            f"{output_path.stem}.generated{output_path.suffix}"
        )

    final_path.parent.mkdir(parents=True, exist_ok=True)
    with final_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns())
        writer.writeheader()
        writer.writerows(rows)
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-json",
        type=Path,
        action="append",
        default=None,
        help=(
            "Phase 3 prompt JSON with instruction/output rows. Can be repeated. "
            "Defaults to the canonical PPO unlabeled prompt train/test JSON files."
        ),
    )
    parser.add_argument(
        "--model-result",
        type=parse_model_result_arg,
        action="append",
        required=True,
        help=(
            "Per-model SFT prediction JSON/JSONL as SHORT=path. Required once "
            "for each of GLM4, GEMMA, LLAMA3, MISTRAL and PHI3."
        ),
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with SEG and TARGET columns. When set, only those "
            "prompts are emitted, in the CSV order."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path. Defaults to data/{DEFAULT_OUTPUT_NAME}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output CSV if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_jsons = args.prompt_json or list(DEFAULT_PROMPT_JSONS)

    model_results = dict(args.model_result)
    missing_models = [short for short in MODEL_ORDER if short not in model_results]
    if missing_models:
        raise SystemExit(
            "Missing --model-result entries for: " + ", ".join(missing_models)
        )

    prompt_rows = apply_selection(load_prompt_rows(prompt_jsons), args.selection_csv)
    predictions_by_model = {
        short: load_model_predictions(path)
        for short, path in model_results.items()
    }
    rows = build_wide_rows(prompt_rows, predictions_by_model)
    output_path = write_csv(rows, args.output, args.overwrite)
    print(f"Wrote {output_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
