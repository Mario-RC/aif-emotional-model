"""Inject negative-quality samples into the filtered comparative data.

For ``n_random_entries`` rows (30% of the data on the train split, 87 on the
test split), drop one of the worst-3 SFT predictions and prepare a
``predict_sft_modified`` entry. Then, on a subset of those rows, replace one
of the three response sentences with a deliberately bad version produced by
an instruct-tuned model:

- Three "(EMOTION)" classes: change one of the three emotion tags.
- "EMPATHY"  – ask the model to rewrite the empathy sentence non-empathetically.
- "EMOTION"  – ask the model to rewrite the emotion sentence with the opposite tone.
- "QUESTION" – ask the model to rewrite the follow-up question as a flat statement.

Outputs are written to ``data/rm_comparison_dataset[_test].json``.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from _lib import read_json, with_suffix, write_json
from tqdm.auto import tqdm

EMOTION_TAGS = ["(ANGER)", "(DISGUST)", "(FEAR)", "(HAPPINESS)", "(NEUTRAL)", "(SADNESS)", "(SURPRISE)"]
DEFAULT_LLAMA_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


@dataclass
class NegativeSampleConfig:
    """Number of rows to mutate per category. Train split values are the defaults."""

    n_random_entries: int = 500
    n_emo_per_position: int = 50  # x3 (emo1, emo2, emo3)
    n_empathy: int = 75
    n_emotion: int = 200
    n_question: int = 75

    @classmethod
    def for_test(cls) -> "NegativeSampleConfig":
        return cls(
            n_random_entries=87,
            n_emo_per_position=9,
            n_empathy=13,
            n_emotion=34,
            n_question=13,
        )


# ---------------------------------------------------------------------------
# Step 1: drop one of the worst predictions for n_random_entries rows.
# ---------------------------------------------------------------------------

def drop_worst_for_random_subset(
    rm_comparison_dataset_filtered: list[dict], cfg: NegativeSampleConfig
) -> list[int]:
    random.seed(42)
    sampled = random.sample(rm_comparison_dataset_filtered, cfg.n_random_entries)
    sampled_idxs = [rm_comparison_dataset_filtered.index(entry) for entry in sampled]

    random.seed(42)
    for offset, idx in tqdm(
        enumerate(sampled_idxs),
        total=len(sampled_idxs),
        desc="1/6 Drop worst sampled rows",
        unit="row",
    ):
        entry = rm_comparison_dataset_filtered[idx]
        scores = entry["scores"]
        bottom_indices = sorted(range(len(scores)), key=lambda i: scores[i])[:3]
        random.seed(offset)
        drop_idx = random.choice(bottom_indices)

        predict_sft_keys = list(entry.keys())[4:13]
        entry.pop(predict_sft_keys[drop_idx])
        entry["scores"].pop(drop_idx)

    # Mark non-sampled rows as untouched.
    non_sampled = list(set(range(len(rm_comparison_dataset_filtered))) - set(sampled_idxs))
    for idx in tqdm(non_sampled, desc="1/6 Mark untouched rows", unit="row"):
        rm_comparison_dataset_filtered[idx].pop("predict_sft_modified", None)
        rm_comparison_dataset_filtered[idx]["predict_sft_modified_label"] = "None"
    return sampled_idxs


# ---------------------------------------------------------------------------
# Step 2: split the target into [emo1, r_empathy, emo2, r_emotion, emo3, r_question]
# ---------------------------------------------------------------------------

def split_emo_utt(response: str) -> list[str]:
    res_ini = [i for i in range(len(response)) if response.startswith("(", i)]
    res_end = [i for i in range(len(response)) if response.startswith(")", i)]
    return [
        response[res_ini[0] : res_end[0] + 1].strip(),
        response[res_end[0] + 1 : res_ini[1]].strip(),
        response[res_ini[1] : res_end[1] + 1].strip(),
        response[res_end[1] + 1 : res_ini[2]].strip(),
        response[res_ini[2] : res_end[2] + 1].strip(),
        response[res_end[2] + 1 :].strip(),
    ]


def change_emotion(current: str) -> str:
    """Pick a random emotion tag distinct from the current one."""
    candidates = list(EMOTION_TAGS)
    random.shuffle(candidates)
    for emotion in candidates:
        if emotion not in current:
            return emotion
    return current


# ---------------------------------------------------------------------------
# Step 3: sample the rows that will receive each kind of mutation.
# ---------------------------------------------------------------------------

def sample_modify_entries(
    candidates: list[str], cfg: NegativeSampleConfig
) -> list[list[str]]:
    pool = list(candidates)
    buckets: list[list[str]] = []
    for _ in tqdm(range(3), desc="2/6 Sample emotion buckets", unit="bucket"):
        bucket = random.sample(pool, cfg.n_emo_per_position)
        pool = [d for d in pool if d not in bucket]
        buckets.append(bucket)

    for size in tqdm((cfg.n_empathy, cfg.n_emotion, cfg.n_question), desc="2/6 Sample LLM buckets", unit="bucket"):
        bucket = random.sample(pool, size)
        pool = [d for d in pool if d not in bucket]
        buckets.append(bucket)
    return buckets


# ---------------------------------------------------------------------------
# Step 4: apply the emotion-tag mutations directly.
# ---------------------------------------------------------------------------

def apply_emotion_mutation(
    data: list[dict], dids: list[str], position_idx: int, label: str
) -> None:
    for did in tqdm(dids, desc=f"3/6 Apply {label} mutations", unit="row"):
        idx = next(i for i, entry in enumerate(data) if entry["did"] == did)
        target = data[idx]["target"]
        target_split = split_emo_utt(target)
        target_split[position_idx] = change_emotion(target_split[position_idx])
        data[idx]["predict_sft_modified"] = " ".join(target_split)
        data[idx]["predict_sft_modified_label"] = label


# ---------------------------------------------------------------------------
# Step 5: apply the LLM-driven mutations (empathy, emotion, question).
# ---------------------------------------------------------------------------

def _build_pipeline(model_id: str, device: int | None = None):
    import torch
    from transformers import pipeline

    resolved_device = device
    if resolved_device is None:
        resolved_device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16 if resolved_device >= 0 else torch.float32,
        device=resolved_device,
    )


def _llm_rewrite(pipe, system: str, user: str, max_new_tokens: int = 512) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return pipe(messages, max_new_tokens=max_new_tokens)[0]["generated_text"][-1]["content"].strip()


def apply_llm_mutation(
    data: list[dict],
    dids: list[str],
    target_position: int,
    label: str,
    pipe,
    system_template: str,
    user_message: str,
) -> None:
    for did in tqdm(dids, desc=f"5/6 Apply {label} LLM mutations", unit="row"):
        idx = next(i for i, entry in enumerate(data) if entry["did"] == did)
        target_split = split_emo_utt(data[idx]["target"])
        system = system_template.format(text=target_split[target_position])
        target_split[target_position] = _llm_rewrite(pipe, system, user_message)
        data[idx]["predict_sft_modified"] = " ".join(target_split)
        data[idx]["predict_sft_modified_label"] = label


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def add_negative_samples(
    is_test: bool = False,
    cfg: NegativeSampleConfig | None = None,
    llama_model: str = DEFAULT_LLAMA_MODEL,
    device: int | None = None,
    skip_llm: bool = False,
) -> None:
    cfg = cfg or (NegativeSampleConfig.for_test() if is_test else NegativeSampleConfig())

    in_path = f"data/{with_suffix('rm_comparison_dataset_filtered', 'json', is_test)}"
    out_path = f"data/{with_suffix('rm_comparison_dataset', 'json', is_test)}"

    data = read_json(in_path)
    drop_worst_for_random_subset(data, cfg)
    write_json(data, out_path)

    data = read_json(out_path)
    candidates = [
        entry["did"]
        for entry in tqdm(data, desc="2/6 Collect mutation candidates", unit="row")
        if "predict_sft_modified" in entry
    ]
    buckets = sample_modify_entries(candidates, cfg)

    apply_emotion_mutation(data, buckets[0], position_idx=0, label="emo1")
    apply_emotion_mutation(data, buckets[1], position_idx=2, label="emo2")
    apply_emotion_mutation(data, buckets[2], position_idx=4, label="emo3")

    if not skip_llm:
        print("4/6 Loading LLM rewrite pipeline...")
        pipe = _build_pipeline(llama_model, device=device)
        apply_llm_mutation(
            data, buckets[3], target_position=1, label="empathy", pipe=pipe,
            system_template="This phrase is empathetic: {text}",
            user_message="Create a sentence that is the opposite, non-empathetic at all.",
        )
        apply_llm_mutation(
            data, buckets[4], target_position=3, label="emotion", pipe=pipe,
            system_template="This phrase has an emotion: {text}",
            user_message="Create a sentence that is the opposite emotion.",
        )
        apply_llm_mutation(
            data, buckets[5], target_position=5, label="question", pipe=pipe,
            system_template="This sample sentence is open-ended to follow a conversation: {text}",
            user_message=(
                "Create sentences that do the opposite, less interesting non-engaging, shorter "
                "questions for each of the examples. Try to use statements instead of questions."
            ),
        )

    print("6/6 Writing final dataset...")
    write_json(data, out_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--llama-model", default=DEFAULT_LLAMA_MODEL)
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="CUDA device index for LLM rewrites. Defaults to first visible GPU, or CPU (-1) if CUDA is unavailable.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the LLM rewrites for debugging; this does not create the full mutated dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    add_negative_samples(
        is_test=args.test,
        llama_model=args.llama_model,
        device=args.device,
        skip_llm=args.skip_llm,
    )
