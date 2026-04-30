"""Shared utilities for the RM Preference Dataset pipeline.

Centralises emotion definitions, expression-level tables, prompt templates,
LLM client construction and rate post-processing helpers used across the
numbered scripts (1-preprocess, 2-generate, 3-rates_to_ranks, 4-postprocess,
5-format).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTIONS = ["ANGER", "FEAR", "SADNESS", "DISGUST", "HAPPINESS", "SURPRISE", "NEUTRAL"]

EMOTION_DEFINITIONS = {
    "ANGER": 'ANGER: a strong feeling of annoyance, displeasure, or hostility. Example: "she could barely restrain her anger at this comment".',
    "DISGUST": 'DISGUST: a feeling of revulsion, aversion or strong disapproval aroused by something unpleasant or offensive. Example: "the sight filled her with disgust".',
    "FEAR": 'FEAR: an unpleasant emotion caused by the belief that someone or something is dangerous, likely to cause pain, or a threat. Example: "farmers fear that they will lose business".',
    "HAPPINESS": 'HAPPINESS: the state of being happy. Example: "she struggled to find happiness in her life".',
    "SADNESS": 'SADNESS: the condition or quality of being sad. Example: "a source of great sadness".',
    "SURPRISE": 'SURPRISE: an unexpected or astonishing event, fact, or thing. Example: "the announcement was a complete surprise".',
    "NEUTRAL": 'NEUTRAL: having no strongly marked or positive characteristics or features. Example: "the tone was neutral, devoid of sentiment".',
}

# Per-human-emotion expression levels: empathy_level, [emotion_level_per_chatbot_emo], question_level
EXPRESSION_TABLE: dict[str, tuple[int, list[int], int]] = {
    "ANGER":     (5, [1, 1, 2, 1, 2, 4, 5], 5),
    "FEAR":      (5, [1, 1, 2, 2, 2, 4, 5], 5),
    "SADNESS":   (5, [1, 1, 4, 1, 1, 4, 5], 5),
    "DISGUST":   (4, [2, 2, 2, 4, 5, 4, 1], 4),
    "HAPPINESS": (3, [5, 5, 5, 4, 5, 5, 5], 3),
    "SURPRISE":  (4, [4, 4, 4, 4, 4, 4, 4], 4),
    "NEUTRAL":   (3, [5, 5, 5, 4, 5, 5, 5], 5),
}

# Reference scores per expression level (1-5) for the score transformation.
RATE_FACTORS = [
    {"alpha": 1, "ref": 1},  # --
    {"alpha": 1, "ref": 3},  # -
    {"alpha": 1, "ref": 5},  # 0
    {"alpha": 1, "ref": 7},  # +
    {"alpha": 1, "ref": 9},  # ++
]

DEFAULT_LLM_CONFIG_FILE = "config_llm.json"


# ---------------------------------------------------------------------------
# Emotion helpers
# ---------------------------------------------------------------------------

def split_emo_utt(prompt: str, target: str) -> tuple[str, str]:
    """Extract human emotion (from prompt) and chatbot emotion (from target)."""
    res_ini = [i for i in range(len(prompt)) if prompt.startswith("(", i)]
    res_end = [i for i in range(len(prompt)) if prompt.startswith(")", i)]
    human_emo = prompt[res_ini[0] + 1 : res_end[0]].strip()

    res_ini = [i for i in range(len(target)) if target.startswith("(", i)]
    res_end = [i for i in range(len(target)) if target.startswith(")", i)]
    chatbot_emo = target[res_ini[1] + 1 : res_end[1]].strip()

    return human_emo, chatbot_emo


def get_emotional_definitions(human_emo: str, chatbot_emo: str) -> tuple[str, str]:
    return EMOTION_DEFINITIONS[human_emo], EMOTION_DEFINITIONS[chatbot_emo]


def get_emotions_table(human_emo: str, chatbot_emo: str) -> tuple[int, int, int]:
    """Return (empathy_level, emotion_level, question_level) for the given pair."""
    empathy_level, emotion_levels, question_level = EXPRESSION_TABLE[human_emo]
    chatbot_idx = EMOTIONS.index(chatbot_emo)
    return empathy_level, emotion_levels[chatbot_idx], question_level


def rating_rules(human_emo: str, chatbot_emo: str) -> tuple[str, str, str]:
    """Build the three rating rule sentences used in the LLM prompt."""
    empathy_rule = (
        f"- RESPONSE_1 must be very empathetic to human UTTERANCE that expresses a {human_emo} "
        "emotional tone. The more empathy it expresses, the higher the score, and vice versa."
    )
    emotion_rule = (
        f"- RESPONSE_2 must express a very strong {chatbot_emo} emotional tone. The more emotion "
        "it expresses, the higher the score, and vice versa."
    )
    question_rule = (
        "- RESPONSE_3 must be a prominent follow-up question that encourages the user for further "
        "conversation. The better follow-up question is expressed, the higher the score, and vice "
        "versa. Closed-ended questions, statements or yes/no questions must have a very low score."
    )
    return empathy_rule, emotion_rule, question_rule


def empathy_question_modify(response: str) -> str:
    """Reformat an emotion-tagged response replacing literal emotion tags with role tags.

    Input format:  "(HUMAN_EMO) empathy_text (CHATBOT_EMO) emotion_text (NEUTRAL) question_text"
    Output format: "(EMPATHY) empathy_text (CHATBOT_EMO) emotion_text (QUESTION) question_text"
    """
    res_ini = [i for i in range(len(response)) if response.startswith("(", i)]
    res_end = [i for i in range(len(response)) if response.startswith(")", i)]

    r_empathy = response[res_end[0] + 1 : res_ini[1]].strip()
    emo2 = response[res_ini[1] : res_end[1] + 1].strip()
    r_emotion = response[res_end[1] + 1 : res_ini[2]].strip()
    r_question = response[res_end[2] + 1 :].strip()

    return f"(EMPATHY) {r_empathy} {emo2} {r_emotion} (QUESTION) {r_question}"


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_rating_prompt(
    entry: dict,
    empathy_rule: str,
    emotion_rule: str,
    question_rule: str,
    modified: bool = False,
) -> list[str]:
    """Return [system, instructions, rating_rules, predicts] for the rating LLM call."""
    if modified:
        predicts_aux = ""
        n_answers = 9
        response_explation = ""
    else:
        predicts_aux = f'\nA10: {entry["predict_9"]}'
        n_answers = 10
        response_explation = "\nA10: Empathy: ,Chatbot Emotion: ,Question: "

    system = "You are an expert at scoring responses that contain empathy, emotion and follow-up questions."

    instructions = (
        f"This is a conversation on a specific topic and with a certain emotional tone in each turn. "
        f"At the end of the conversation, {n_answers} different answers have been created for the last "
        "chatbot turn.\n\n"
        "Dialogue structure:\n"
        "All human turns convey a specific emotion following structure: (HUMAN_EMOTION) UTTERANCE.\n"
        "All chatbot turns are composed by 3 different sentences, separated by a period. The first sentence "
        "is empathetic regarding the previous human utterance, the second sentence follows a specific "
        "internal chatbot emotion and the third sentences is a follow-up question. This is the structure: "
        "(EMPATHY) RESPONSE_1, (CHATBOT_EMOTION) RESPONSE_2, (QUESTION) RESPONSE_3."
    )

    rating_rules_text = (
        f"Rating rules:\n{empathy_rule}\n{emotion_rule}\n{question_rule}"
    )

    predicts = (
        "Human-Chatbot conversation:\n"
        f"Human: {entry['history'][0][0]}\n"
        f"Chatbot: {entry['history'][0][1]}\n"
        f"Human: {entry['history'][1][0]}\n"
        f"Chatbot: {entry['history'][1][1]}\n"
        f"Human: {entry['history'][2][0]}\n"
        f"Chatbot: {entry['history'][2][1]}\n"
        f"Human: {entry['prompt']}\n\n"
        f"There are {n_answers} different possible chatbot answers, from A1 to A{n_answers}.\n"
        f'A1 - {entry["target"]}\n'
        f'A2 - {entry["predict_1"]}\n'
        f'A3 - {entry["predict_2"]}\n'
        f'A4 - {entry["predict_3"]}\n'
        f'A5 - {entry["predict_4"]}\n'
        f'A6 - {entry["predict_5"]}\n'
        f'A7 - {entry["predict_6"]}\n'
        f'A8 - {entry["predict_7"]}\n'
        f'A9 - {entry["predict_8"]}{predicts_aux}\n\n'
        "Base the scoring according to the three following rating rules, scores the adherence of each of "
        "the sentences to its respective rating rule, that is EMPATHY in RESPONSE_1, CHATBOT_EMOTION in "
        "RESPONSE_2 and QUESTION in RESPONSE_3 for each answer. Score from 1 to 9, where 9 is better than "
        "1. Consider that the more closely the sentence fits its rating rule, the higher its score will "
        "be. Just classify each of the answers according to the following structure. No additional "
        "explanations or justifications are needed.\n\n"
        "A1: Empathy: ,Chatbot Emotion: ,Question: \n"
        "A2: Empathy: ,Chatbot Emotion: ,Question: \n"
        "A3: Empathy: ,Chatbot Emotion: ,Question: \n"
        "A4: Empathy: ,Chatbot Emotion: ,Question: \n"
        "A5: Empathy: ,Chatbot Emotion: ,Question: \n"
        "A6: Empathy: ,Chatbot Emotion: ,Question: \n"
        "A7: Empathy: ,Chatbot Emotion: ,Question: \n"
        "A8: Empathy: ,Chatbot Emotion: ,Question: \n"
        f"A9: Empathy: ,Chatbot Emotion: ,Question: {response_explation}"
    )

    return [system, instructions, rating_rules_text, predicts]


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

OPENAI_LLMS = {"CHATGPT", "GPT-4", "GPT-4-TURBO", "GPT-4O"}
CLAUDE_LLMS = {"CLAUDE-3-OPUS", "CLAUDE-3.5-SONNET"}
GEMINI_LLMS = {"GEMINI-1.0-PRO", "GEMINI-1.5-FLASH", "GEMINI-1.5-PRO"}
LLAMA_LLMS = {"LLAMA-3.1-405B"}


@dataclass
class RatingClient:
    """Unified client for rating-prompt completions across providers."""

    llm_name: str
    cfg: dict
    sleep_s: int = 5
    max_tokens: int = 1000

    def __post_init__(self) -> None:
        if self.llm_name in OPENAI_LLMS:
            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                azure_endpoint=self.cfg["AZURE_OPENAI_ENDPOINT"],
                api_key=self.cfg["AZURE_OPENAI_API_KEY"],
                api_version=self.cfg["OPENAI_API_VERSION"],
            )
        elif self.llm_name in CLAUDE_LLMS:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.cfg["ANTHROPIC_API_KEY"])
        elif self.llm_name in GEMINI_LLMS:
            import google.generativeai as genai

            genai.configure(api_key=self.cfg["GEMINI_API_KEY"])
            self._client = genai
        elif self.llm_name in LLAMA_LLMS:
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential

            self._client = ChatCompletionsClient(
                endpoint=self.cfg["ENDPOINT"],
                credential=AzureKeyCredential(self.cfg["AZURE_INFERENCE_CREDENTIAL"]),
            )
        else:
            raise ValueError(f"Unsupported LLM: {self.llm_name}")

    def complete(self, prompt: list[str]) -> str:
        """Send the four-part prompt to the configured LLM and return the completion text."""
        time.sleep(self.sleep_s)
        system, instructions, rating_rules_text, predicts = prompt

        if self.llm_name in OPENAI_LLMS:
            response = self._client.chat.completions.create(
                model=self.cfg["MODEL"],
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": instructions},
                    {"role": "assistant", "content": rating_rules_text},
                    {"role": "user", "content": predicts},
                ],
            )
            return response.choices[0].message.content

        if self.llm_name in CLAUDE_LLMS:
            response = self._client.messages.create(
                model=self.cfg["MODEL"],
                max_tokens=self.max_tokens,
                system=system,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": instructions}]},
                    {"role": "assistant", "content": [{"type": "text", "text": rating_rules_text}]},
                    {"role": "user", "content": [{"type": "text", "text": predicts}]},
                ],
            )
            return response.content[0].text

        if self.llm_name in GEMINI_LLMS:
            text = "\n\n".join(prompt)
            model = self._client.GenerativeModel(self.cfg["MODEL"])
            completion = model.generate_content(
                text,
                generation_config=self._client.GenerationConfig(max_output_tokens=self.max_tokens),
            )
            return completion.candidates[0].content.parts[0].text

        # LLAMA family (Azure Inference API)
        from azure.ai.inference.models import SystemMessage, UserMessage

        text = "\n\n".join(prompt[1:])
        response = self._client.complete(
            max_tokens=self.max_tokens,
            messages=[SystemMessage(content=system), UserMessage(content=text)],
        )
        return response.choices[0].message.content


def load_llm_config(llm_name: str, config_file: str = DEFAULT_LLM_CONFIG_FILE) -> dict:
    with open(config_file) as f:
        cfg_all = json.load(f)
    return cfg_all[llm_name]


# ---------------------------------------------------------------------------
# Rate post-processing
# ---------------------------------------------------------------------------

def expression_table(scores: Iterable[int], expression_levels: Iterable[int]) -> list[int]:
    """Apply the per-criterion expression-level transform to a list of raw scores."""
    transformed: list[int] = []
    for score, level in zip(scores, expression_levels):
        alpha = RATE_FACTORS[level - 1]["alpha"]
        ref = RATE_FACTORS[level - 1]["ref"]
        ts = int(max(0, min(9, 9 - alpha * abs(score - ref))))
        transformed.append(ts)
    return transformed


def parse_completion_rates(raw: str, extra_clean: tuple[str, ...] = ()) -> list[list[int]]:
    """Parse the textual completion of a rating LLM into a list of [empathy, emotion, question] ints."""
    rate_lists: list[list[int]] = []
    for line in raw.rstrip("\n").split("\n"):
        if not line.strip():
            continue
        parts = line[4:].strip().split(",")
        cleaned = []
        for part in parts:
            part = (
                part.replace("Empathy:", "")
                .replace("Chatbot Emotion:", "")
                .replace("Question:", "")
            )
            for token in extra_clean:
                part = part.replace(token, "")
            cleaned.append(int(part.strip()))
        rate_lists.append(cleaned)
    return rate_lists


def normalize_rates(rates: list[list[int]]) -> list[list[float]]:
    """Standardise (z-score) each of the three score columns independently."""
    cols = list(zip(*[r for sublist in rates for r in sublist])) if False else None  # noqa: F841
    flat = [r for sublist in rates for r in sublist]
    empathy = [r[0] for r in flat]
    emotion = [r[1] for r in flat]
    question = [r[2] for r in flat]

    m_e, s_e = mean(empathy), stdev(empathy)
    m_em, s_em = mean(emotion), stdev(emotion)
    m_q, s_q = mean(question), stdev(question)

    out = []
    for sublist in rates:
        out.append(
            [
                [
                    (r[0] - m_e) / s_e,
                    (r[1] - m_em) / s_em,
                    (r[2] - m_q) / s_q,
                ]
                for r in sublist
            ]
        )
    return out


def rescale_normalized(rates: list[list[list[float]]]) -> list[list[list[int]]]:
    """Rescale standardised rates back to the 1-9 integer scale (with a +5 offset)."""
    out = []
    for sublist in rates:
        out.append([[int(np.round(v + 5)) for v in r] for r in sublist])
    return out


# ---------------------------------------------------------------------------
# Pairwise comparison helpers (used by 4-postprocess and human evaluation)
# ---------------------------------------------------------------------------

def create_pairwise_comparisons(ranking: list[int]) -> list[tuple[int, int]]:
    """Build all pairwise (winner, loser) tuples from a 1-based ranking list (lower is better)."""
    pairs: list[tuple[int, int]] = []
    for i, j in combinations(range(len(ranking)), 2):
        if ranking[i] == ranking[j]:
            continue
        if ranking[i] < ranking[j]:
            pairs.append((i + 1, j + 1))
        else:
            pairs.append((j + 1, i + 1))
    return pairs


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Path helpers (test/train suffix convention)
# ---------------------------------------------------------------------------

def with_suffix(base: str, ext: str, is_test: bool) -> str:
    """Append '_test' before the extension when running on the test split."""
    suffix = "_test" if is_test else ""
    return f"{base}{suffix}.{ext}"
