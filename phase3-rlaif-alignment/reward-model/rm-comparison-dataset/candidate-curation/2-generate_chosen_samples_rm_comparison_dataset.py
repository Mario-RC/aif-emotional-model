"""Build a "choose-the-best" instruction file for each model.

For each rm_comparison_dataset_results entry the seven candidate predictions
(``predict_sft_1`` ... ``predict_sft_7``) are concatenated and wrapped in an
instruction asking the model to pick the response that best follows the
emotional/dialogue rules. The output is written to
``data/<model>/chosen_response_<short>.json`` for each of the five SFT models.

NOTE: marked "DON'T USE" in the original notebook. Kept for historical reference.
"""

from __future__ import annotations

from _lib import MODELS, MODEL_TO_NAME, read_json, write_json

PREDICT_SFT_KEYS = [f"predict_sft_{i}" for i in range(1, 8)]

CHOOSER_SYSTEM_PROMPT = """You are a chatbot capable of choosing the best response to a user's sentence.

The turns of the conversation have the following structure.
Human: (prompt_emotion) PROMPT.
Chatbot: (prompt_emotion) RESPONSE_1. (chatbot_emotion) RESPONSE_2. (NEUTRAL) RESPONSE_3.

Dialogue rules:
Create an open-domain curated dialogue between two humans. The dialogue should be coherent, engaging and proactive.
Between RESPONSE_1, RESPONSE_2 and RESPONSE_3 must be a max length of 20-25 words.
RESPONSE_3 must be open-ended to follow-up the conversation, so the Human is encouraged to answer with a full long sentence. Avoid yes/no questions.

Emotional rules:
Remember these different emotions: ANGER, DISGUST, FEAR, HAPPINESS, SADNESS, SURPRISE and NEUTRAL.
PROMPT and RESPONSE_1 must be empathetic to each other.
RESPONSE_2 in each turn must show a lot of emotion and not be at all empathetic to Human.
RESPONSE_3 must contain the NEUTRAL tone.

Emotional definitions:
ANGER: a strong feeling of annoyance, displeasure, or hostility. Example: she could barely restrain her anger at this comment.
DISGUST: a feeling of revulsion, aversion or strong disapproval aroused by something unpleasant or offensive. Example: the sight filled her with disgust.
FEAR: an unpleasant emotion caused by the belief that someone or something is dangerous, likely to cause pain, or a threat. Example: farmers fear that they will lose business.
HAPPINESS: the state of being happy. Example: she struggled to find happiness in her life.
SADNESS: the condition or quality of being sad. Example: a source of great sadness.
SURPRISE: an unexpected or astonishing event, fact, or thing. Example: the announcement was a complete surprise.
NEUTRAL: having no strongly marked or positive characteristics or features. Example: the tone was neutral, devoid of sentiment.

Choose the best sentence among these 7 responses that follow exactly all the dialogues and emotional rules, emotional definitions, emotional structure:"""


def _build_chooser_record(entry: dict) -> dict:
    candidate_responses = "\n".join(entry[k] for k in PREDICT_SFT_KEYS)
    return {
        "system": CHOOSER_SYSTEM_PROMPT,
        "instruction": candidate_responses,
        "input": "",
        "model": entry["model"],
        "did": entry["did"],
    }


def build_chooser_dataset() -> None:
    for model in MODELS:
        data = read_json(
            f"../llama-factory-predict/saves/{model}/emotional_balanced/rm_comparison_dataset_results.json"
        )
        chooser_entries = [_build_chooser_record(entry) for entry in data]
        write_json(
            chooser_entries,
            f"data/{model}/chosen_response_{MODEL_TO_NAME[model]}.json",
        )


if __name__ == "__main__":
    build_chooser_dataset()
