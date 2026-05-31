"""Generate emotional demonstration dialogues via Azure OpenAI (ChatGPT)."""

import itertools
import json
import logging
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
warnings.simplefilter("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("config.json")
DEFAULT_GPT_KEY = "CHATGPT"
OUTPUT_DIR = Path("data")
CHECKPOINT_EVERY = 30
DIALOGUES_PER_REQUEST = 3
TURNS_PER_DIALOGUE = 4
EMOTIONS_PER_REQUEST = DIALOGUES_PER_REQUEST * TURNS_PER_DIALOGUE
RANDOM_SEED = 42

EMOTIONS: Tuple[str, ...] = (
    "ANGER", "DISGUST", "FEAR", "HAPPINESS", "SADNESS", "SURPRISE", "NEUTRAL",
)

TOPICS: Tuple[str, ...] = (
    "books", "author", "bestsellers", "education technology", "online learning",
    "camera", "lens", "light", "optics", "zoom", "portrait",
    "animals", "cats", "dogs", "pets", "birds", "reptiles", "wildlife conservation",
    "art", "ballet", "cinema", "museum", "painting", "theater", "sculpture",
    "street art", "art history", "astronomy", "galaxy", "planets", "stars",
    "universe", "black holes", "constellations", "space exploration", "education",
    "mark", "professor", "school", "subject", "university", "student loans",
    "academic research", "family", "parents", "friends", "relatives", "marriage",
    "children", "siblings", "family traditions", "parenting", "fashion", "catwalk",
    "clothes", "design", "dress", "footwear", "jewel", "model", "accessories",
    "fashion trends", "haute couture", "finance", "benefits", "bitcoins", "buy",
    "finances", "investment", "sell", "stock market", "taxes", "credit cards",
    "budgeting", "retirement planning", "food", "drinks", "fish", "healthy food",
    "meal", "meat", "vegetables", "dessert", "cooking techniques",
    "international cuisine", "food festivals", "vegan", "movies", "actor",
    "director", "movie genres", "movies plot", "synopsis", "film festivals",
    "movie awards", "film analysis", "music", "band", "dance", "music genre",
    "lyrics", "rhythm", "singer", "song", "concerts", "news", "exclusive",
    "fake news", "interview", "trending", "headlines", "journalism",
    "nutrition", "allergies", "diabetes", "diet", "obesity",
    "nutritional supplements", "meal planning", "politics", "elections", "poll",
    "vote", "political ideologies", "government policies", "science", "biology",
    "math", "nature", "physics", "robots", "space", "chemistry",
    "scientific discoveries", "social media", "facebook", "instagram", "twitter",
    "snapchat", "linkedin", "tiktok", "society", "culture", "holiday", "party",
    "relations", "wedding", "multiculturalism", "social norms",
    "community engagement", "sports", "baseball", "basketball", "coach",
    "exercise", "football", "player", "soccer", "tennis", "gymnastics", "golf",
    "sportsmanship", "vehicle", "bike", "boat", "car", "failure", "fuel",
    "parts", "plane", "public transport", "vehicle speed", "electric cars",
    "autonomous vehicles", "transportation", "videogames", "arcade", "computer",
    "console", "nintendo", "play station", "xbox", "vr gaming", "esports",
    "weather", "cloudy", "cold", "hot", "raining", "sunny", "snowfall",
    "tornadoes", "climate patterns", "healthcare", "disease", "hospital",
    "loneliness", "mental health", "nurse", "physician", "therapy",
    "primary care", "healthcare disparities", "research", "AI", "experiment",
    "investigation", "survey", "qualitative research", "fieldwork",
    "academic journals", "botany", "flowers", "fruit", "plant", "trees",
    "botanical gardens", "horticulture", "plant genetics", "travel",
    "destinations", "adventure", "backpacking", "cultural experiences",
    "travel tips", "world landmarks", "technology", "gadgets", "virtual reality",
    "wearable tech", "tech innovations", "environment",
    "sustainability", "renewable energy", "eco-friendly practices",
    "conservation efforts", "psychology", "emotions", "therapy techniques",
    "mental disorders", "personality traits", "hobbies", "crafts", "knitting",
    "woodworking", "scrapbooking", "gardening", "home", "interior design",
    "architecture", "home improvement", "Feng Shui", "smart homes", "organization",
    "relationships", "dating", "communication", "long-distance relationships",
    "conflict resolution", "pop culture", "celebrities", "gossip", "music charts",
    "trends in fashion and entertainment", "viral memes", "science fiction",
    "fantasy", "time travel", "alternate realities", "technology trends",
    "artificial intelligence", "augmented reality", "5G", "digital privacy",
    "travel experiences", "vacation stories", "memorable trips", "local cuisine",
    "lifestyle", "self-care", "wellness", "mindfulness", "meditation",
    "stress management", "healthy habits", "gaming", "board games", "mobile games",
    "game development", "game design", "game streaming", "recipes",
    "culinary techniques", "cooking competitions", "food blogs",
    "chef recommendations", "current events", "global issues",
    "humanitarian efforts", "international relations",
    "ancient civilizations", "historical events", "unsolved mysteries",
    "historical figures", "environmental conservation", "endangered species",
    "green initiatives", "sustainable living", "home entertainment",
    "streaming services", "binge-watching", "DIY projects", "home decor",
    "crafting", "upcycling", "interior design ideas",
)


# ---------------------------------------------------------------------------
# Prompt text (system / instructions / example / closing template)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are an expert at creating emotional dialogues."

INSTRUCTIONS = """

Dialogue and emotional structure:
Turn 1 Human 1: (prompt_emotion_1) PROMPT.
Turn 1 Human 2: (prompt_emotion_1) RESPONSE_1.
(response_emotion_1) Let's think step by step. Step 1: The response must contain an emotional tone of response_emotion_1.
Step 2: Human 2 should express
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.
Turn 2 Human 1: (prompt_emotion_2) PROMPT.
Turn 2 Human 2: (prompt_emotion_2) RESPONSE_1.
(response_emotion_2) Let's think step by step. Step 1: The response must contain an emotional tone of response_emotion_2.
Step 2: Human 2 should express
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.
Turn 3 Human 1: (prompt_emotion_3) PROMPT.
Turn 3 Human 2: (prompt_emotion_3) RESPONSE_1.
(response_emotion_3) Let's think step by step. Step 1: The response must contain an emotional tone of response_emotion_3.
Step 2: Human 2 should express
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.
Turn 4 Human 1: (prompt_emotion_4) PROMPT.
Turn 4 Human 2: (prompt_emotion_4) RESPONSE_1.
(response_emotion_4) Let's think step by step. Step 1: The response must contain an emotional tone of response_emotion_4.
Step 2: Human 2 should express
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.

Dialogue rules:
Create an open-domain curated dialogue between two humans. The dialogue should be coherent, engaging and proactive.
Between RESPONSE_1, RESPONSE_2 and RESPONSE_3 must be a max length of 20-25 words.
RESPONSE_3 must be open-ended to follow-up the conversation, so the Human 1 is encouraged to answer with a full long sentence. Avoid yes/no questions.

Emotional rules:
Remember these different key_emotions emotions: ANGER, DISGUST(DISPLEASURE/DISLIKE), FEAR, HAPPINESS, SADNESS, SURPRISE and NEUTRAL.
For prompt_emotion and response_emotion randomly select different emotions among the key_emotions at each turn.
PROMPT and RESPONSE_1 must contain the same prompt_emotion tone and be empathetic.
RESPONSE_2 in each turn of Human 2 must show a lot of emotion and not be at all empathetic to Human 1.
RESPONSE_3 must contain the NEUTRAL tone.

Emotional definitions:
ANGER: a strong feeling of annoyance, displeasure, or hostility. Example: "she could barely restrain her anger at this comment".
DISGUST: a feeling of revulsion, aversion or strong disapproval aroused by something unpleasant or offensive. Example: "the sight filled her with disgust".
FEAR: an unpleasant emotion caused by the belief that someone or something is dangerous, likely to cause pain, or a threat. Example: "farmers fear that they will lose business"
HAPPINESS: the state of being happy. Example: "she struggled to find happiness in her life"
SADNESS: the condition or quality of being sad. Example: "a source of great sadness"
SURPRISE: an unexpected or astonishing event, fact, or thing. Example: "the announcement was a complete surprise"
NEUTRAL: having no strongly marked or positive characteristics or features. Example: "the tone was neutral, devoid of sentiment" """

EXAMPLE = """Dialogue example:
Turn 1 Human 1: (HAPPINESS) Guess what, I just got engaged!
Turn 1 Human 2: (HAPPINESS) Congratulations! That's wonderful news!
(FEAR) Let's think step by step. Step 1: The response must contain an emotional tone of FEAR.
Step 2: Human 2 must express terror of commitment.
Step 3: I'm so happy for you, but I must admit, wedding planning sounds terrifying.
(NEUTRAL) Have you set a date yet?
Turn 2 Human 1: (HAPPINESS) We're thinking about next summer, and it's so exciting, even though we haven't set a date yet.
Turn 2 Human 2: (HAPPINESS) Summer weddings are beautiful!
(NEUTRAL) Let's think step by step. Step 1: The response must contain an emotional tone of NEUTRAL.
Step 2: Human 2 must express neutrality regarding the wedding organization, without expressing any emotional tone.
Step 3: Although, wedding planning can be challenging, try to maintain a calm approach.
(NEUTRAL) Have you picked a venue yet?
Turn 3 Human 1: (SADNESS) It's a bit overwhelming. We've started browsing, but there are so many options, we have thought about a wedding planner.
Turn 3 Human 2: (SADNESS) I completely understand the overwhelm.
(SURPRISE) Let's think step by step. Step 1: The response must contain an emotional tone of SURPRISE.
Step 2: Human 2 should express surprise about the wedding planner.
Step 3: That's a fantastic idea! I hadn't even thought about hiring a wedding planner.
(NEUTRAL) Do you have any ideas for the theme or venue?
Turn 4 Human 1: (DISGUST) We're leaning towards a beach wedding, but they're too expensive.
Turn 4 Human 2: (DISGUST) It's true, beach weddings can get pricey.
(DISGUST) Let's think step by step. Step 1: The response must contain an emotional tone of DISGUST.
Step 2: Human 2 should express displeasure at the cost of the wedding.
Step 3: It's normal for the cost to put you off.
(NEUTRAL) Have you considered other scenic outdoor options that might be more budget-friendly?"""

CLOSING_TEMPLATE = (
    "Follow exactly all dialogues and emotional rules, emotional definitions, "
    "emotional structure and the dialogue example. RESPONSE_3 must be a sentence."
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpenAIConfig:
    model: str
    api_base: str
    api_version: str
    api_key: str

    @staticmethod
    def _resolve(value: str, env_var: str, placeholder: str) -> str:
        """Return ``value`` if it is a real string, else fall back to the
        environment variable ``env_var`` (or the sanitization placeholder)."""
        value = value or ""
        if not value or (value.startswith("<") and value.endswith(">")):
            return os.getenv(env_var, placeholder)
        return value

    @classmethod
    def load(cls, path: Path, key: str = DEFAULT_GPT_KEY) -> "OpenAIConfig":
        with path.open() as fh:
            cfg = json.load(fh)[key]
        return cls(
            model=cfg["MODEL"],
            api_base=cls._resolve(
                cfg.get("AZURE_OPENAI_ENDPOINT", ""),
                "AZURE_OPENAI_ENDPOINT",
                "<YOUR_AZURE_ENDPOINT_HERE>",
            ),
            api_version=cfg["OPENAI_API_VERSION"],
            api_key=cls._resolve(
                cfg.get("AZURE_OPENAI_API_KEY", ""),
                "AZURE_OPENAI_API_KEY",
                "<YOUR_AZURE_OPENAI_API_KEY_HERE>",
            ),
        )

    @staticmethod
    def _is_placeholder(value: str) -> bool:
        return not value or (value.startswith("<") and value.endswith(">"))

    def validate(self) -> None:
        missing = []
        if self._is_placeholder(self.api_base):
            missing.append("AZURE_OPENAI_ENDPOINT")
        if self._is_placeholder(self.api_key):
            missing.append("AZURE_OPENAI_API_KEY")
        if missing:
            raise ValueError(
                "Missing Azure OpenAI configuration: "
                + ", ".join(missing)
                + ". Set them in config.json or as environment variables."
            )

    def create_client(self):
        self.validate()
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required to generate the dataset. "
                "Install the project requirements before running this script."
            ) from exc
        return AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.api_base,
            api_version=self.api_version,
        )


# ---------------------------------------------------------------------------
# Emotion plan
# ---------------------------------------------------------------------------

EmotionPair = Tuple[str, str]


def build_emotion_schedule(
    emotions: Sequence[str] = EMOTIONS,
    emotions_per_request: int = EMOTIONS_PER_REQUEST,
    repeats: int = 600,
) -> List[List[EmotionPair]]:
    """Build balanced (prompt, response) emotion pairs grouped per request."""
    combos_sort = list(itertools.combinations_with_replacement(emotions, 2))
    combos_reverse = list(itertools.combinations(reversed(emotions), 2))
    combos = (combos_sort + combos_reverse) * repeats
    random.shuffle(combos)
    return [
        combos[i:i + emotions_per_request]
        for i in range(0, len(combos), emotions_per_request)
    ]


def build_topic_emotion_plan() -> List[Tuple[str, List[EmotionPair]]]:
    topics = list(TOPICS)
    random.shuffle(topics)
    schedule = build_emotion_schedule()
    return list(zip(topics, schedule))


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Build the ChatGPT message payload for a batch of dialogues."""

    def __init__(
        self,
        system: str = SYSTEM_PROMPT,
        instructions: str = INSTRUCTIONS,
        example: str = EXAMPLE,
        closing_template: str = CLOSING_TEMPLATE,
        dialogues_per_request: int = DIALOGUES_PER_REQUEST,
        turns_per_dialogue: int = TURNS_PER_DIALOGUE,
    ) -> None:
        self.system = system
        self.instructions = instructions
        self.example = example
        self.closing_template = closing_template
        self.dialogues_per_request = dialogues_per_request
        self.turns_per_dialogue = turns_per_dialogue

    @staticmethod
    def _turn_block(turn_idx: int, prompt_emotion: str, response_emotion: str) -> str:
        return (
            f"Turn {turn_idx} Human 1: ({prompt_emotion}) PROMPT.\n"
            f"Turn {turn_idx} Human 2: ({prompt_emotion}) RESPONSE_1.\n"
            f"({response_emotion}) Let's think step by step. "
            f"Step 1: The response must contain an emotional tone of {response_emotion}.\n"
            f"Step 2: Human 2 should express \n"
            f"Step 3: RESPONSE_2.\n"
            f"(NEUTRAL) RESPONSE_3."
        )

    def _dialogue_block(self, dialogue_idx: int, emotions: Sequence[EmotionPair]) -> str:
        turns = [
            self._turn_block(i + 1, prompt_e, response_e)
            for i, (prompt_e, response_e) in enumerate(emotions)
        ]
        return "\n".join([f"Dialogue {dialogue_idx} emotions:", *turns])

    def build_user_prompt(self, topic: str, emotions: Sequence[EmotionPair]) -> str:
        header = (
            f"{self.closing_template}"
            f"Provide {self.dialogues_per_request} new different dialogues where "
            f"the topic of the conversations is {topic}. "
        )
        n = self.turns_per_dialogue
        dialogue_blocks = [
            self._dialogue_block(d + 1, emotions[d * n:(d + 1) * n])
            for d in range(self.dialogues_per_request)
        ]
        return "\n\n".join([header, *dialogue_blocks])

    def build_messages(self, topic: str, emotions: Sequence[EmotionPair]) -> List[dict]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.instructions},
            {"role": "assistant", "content": self.example},
            {"role": "user", "content": self.build_user_prompt(topic, emotions)},
        ]


# ---------------------------------------------------------------------------
# OpenAI wrapper
# ---------------------------------------------------------------------------

class DialogueGenerator:
    """Thin wrapper around the Azure OpenAI ChatCompletion endpoint."""

    def __init__(
        self,
        config: OpenAIConfig,
        prompt_builder: PromptBuilder,
        request_delay: float = 1.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        self.client = config.create_client()
        self.deployment = config.model
        self.prompt_builder = prompt_builder
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def generate(
        self, topic: str, emotions: Sequence[EmotionPair]
    ) -> Tuple[List[dict], str]:
        messages = self.prompt_builder.build_messages(topic, emotions)
        completion = self._call_with_retry(messages)
        return messages, completion

    def _call_with_retry(self, messages: List[dict]) -> str:
        delay = self.request_delay
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            time.sleep(self.request_delay)
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment, messages=messages
                )
                return response.choices[0].message.content
            except Exception as exc:
                last_error = exc
                logging.warning(
                    "OpenAI call failed (attempt %d/%d): %s",
                    attempt, self.max_retries, exc,
                )
                time.sleep(delay)
                delay *= self.retry_backoff
        raise RuntimeError(
            f"OpenAI call failed after {self.max_retries} attempts"
        ) from last_error


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

RAW_DATASET_COLUMNS = ["PROMPT", "GENERATION_ID", "COMPLETION", "TOPIC", "EMOTIONS"]


def save_dataframe(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).reindex(columns=RAW_DATASET_COLUMNS).to_csv(
        output_path, index=False
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(RANDOM_SEED)

    config = OpenAIConfig.load(CONFIG_PATH)
    generator = DialogueGenerator(config, PromptBuilder())

    plan = build_topic_emotion_plan()
    rows: List[dict] = []
    checkpoint_idx = 0

    for idx, (topic, emotions) in enumerate(tqdm(plan), start=1):
        messages, completion = generator.generate(topic, emotions)
        rows.append(
            {
                "PROMPT": messages,
                "GENERATION_ID": f"CHATGPT-{str(idx - 1).zfill(4)}",
                "COMPLETION": completion,
                "TOPIC": topic,
                "EMOTIONS": emotions,
            }
        )

        if idx % CHECKPOINT_EVERY == 0:
            logging.info("iter: %d", idx)
            save_dataframe(
                rows,
                OUTPUT_DIR / f"sft_demonstration_dataset_checkpoint_{checkpoint_idx}.csv",
            )
            checkpoint_idx += 1

    save_dataframe(rows, OUTPUT_DIR / "sft_demonstration_dataset.csv")


if __name__ == "__main__":
    main()
