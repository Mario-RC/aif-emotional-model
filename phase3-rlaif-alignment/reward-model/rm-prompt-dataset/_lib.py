"""Shared utilities for the RM Prompt Dataset pipeline.

Holds the topic list, the emotion catalogue, the dialogue prompt template and
small helpers used across the four numbered scripts.
"""

from __future__ import annotations

import itertools
import json
import random
from math import ceil
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTIONS = ["ANGER", "DISGUST", "FEAR", "HAPPINESS", "SADNESS", "SURPRISE", "NEUTRAL"]
DATASET_SET = "rm-prompt"

TOPICS: list[str] = [
    "sunny", "healthy habits", "time management", "social media", "relatives",
    "sports analysis", "sports teams", "haute couture", "constellations", "knitting",
    "historical events", "song", "loneliness", "university", "economic indicators",
    "friends", "investing", "music", "viral memes", "culinary techniques",
    "garden design", "market trends", "board games", "play station", "backpacking",
    "diy furniture", "qualitative research", "reptiles", "physics", "crypto mining",
    "mental health", "failure", "blockchain", "dance", "time travel", "music charts",
    "child development", "horticulture", "physician", "footwear", "nurse", "baseball",
    "climate policies", "parents", "metaphysics", "artificial intelligence", "survey",
    "botanical gardens", "destinations", "pet training", "literary analysis",
    "parenting", "virtual reality", "wellness routines", "ballet", "skincare",
    "science", "famous architects", "healthcare disparities", "space telescopes",
    "journalism", "party", "author", "fruit", "conservation efforts", "world landmarks",
    "tech news", "renewable energy", "tech trends", "holiday", "paragliding",
    "tech startups", "bestsellers", "financial planning", "cognitive psychology",
    "climate patterns", "technology", "real estate", "resume writing", "tech reviews",
    "art", "telescopes", "investigation", "ancient civilizations", "dating",
    "movie awards", "cooking competitions", "linkedin", "technology trends", "diet",
    "historical figures", "cooking techniques", "vegan", "actor", "car safety",
    "writing", "vehicle", "community development", "portrait", "school", "startups",
    "nature", "obesity", "historical artifacts", "physical health",
    "scientific discoveries", "multiculturalism", "nutrition", "space exploration",
    "economics", "business strategies", "flowers", "global warming",
    "trends in fashion and entertainment", "endangered species", "exclusive", "jewel",
    "golf", "animals", "language learning", "mindfulness techniques",
    "academic research", "global issues", "conservation", "education", "health",
    "retirement planning", "credit management", "streaming services", "crop rotation",
    "architectural styles", "workplace skills", "headlines", "zoom", "world wars",
    "pet adoption", "books", "vehicle speed", "sustainable living", "home improvement",
    "philosophy", "job search", "movies plot", "universe", "basketball", "poverty",
    "news", "food", "gossip", "science education", "humanitarian efforts",
    "scientific experiments", "current events", "5g", "xbox", "home maintenance",
    "bungee jumping", "cultural exchange", "band", "computer", "fashion",
    "cultural experiences", "photography", "tech innovations", "interior design",
    "sports", "literary genres", "sports news", "business", "trees", "soccer",
    "altcoins", "fuel", "gymnastics", "nature photography", "wedding",
    "historical architecture", "automotive", "global economy", "beauty",
    "alternate realities", "carbon footprint", "birds", "drawing",
    "wildlife conservation", "planets", "galaxy", "book reviews", "skydiving",
    "healthcare", "communication", "career", "diy projects", "children",
    "famous philosophers", "social justice", "rock climbing", "credit cards",
    "dessert", "mindfulness apps", "psychology", "experiment", "personal training",
    "diabetes", "interview tips", "organic farming", "makeup", "travel photography",
    "car maintenance", "singer", "coach", "ethics", "finances", "vegetables",
    "vr gaming", "cloudy", "science fiction", "travel blogs", "local cuisine",
    "ecosystems", "film analysis", "culture", "relationships", "transportation",
    "primary care", "pet behavior", "sports history", "education technology",
    "stock market", "director", "space missions", "binge-watching", "allergies",
    "raining", "decentralized finance", "personal growth", "automotive technology",
    "electric cars", "contemporary literature", "snapchat", "tennis", "upcycling",
    "emotions", "pottery", "cultural diversity", "haircare", "literature",
    "philosophical theories", "organization", "football", "interior design ideas",
    "climate activism", "food festivals", "tiktok", "fantasy", "scuba diving",
    "space stations", "street art", "marriage", "behavioral psychology", "linguistics",
    "optics", "green initiatives", "concerts", "feng shui", "human rights",
    "electric vehicles", "indoor gardening", "investment", "meal planning",
    "fieldwork", "healthy food", "psychological disorders", "academic journals",
    "smart homes", "history", "urban planning", "developmental psychology", "dogs",
    "festivals", "woodworking", "agricultural technology", "politics", "arcade",
    "pet care", "game design", "sportsmanship", "taxes", "family traditions",
    "student loans", "museum", "architecture", "buy", "community engagement", "mark",
    "travel", "fitness challenges", "catwalk", "pop culture", "facebook",
    "personal finance", "mental disorders", "vote", "car", "relations", "meal",
    "siblings", "family activities", "therapy", "language exchange", "film festivals",
    "gaming", "public transport", "wellness", "fitness", "pet health", "meditation",
    "economic policies", "cooking", "chemistry", "therapy techniques",
    "travel destinations", "agriculture", "mutual funds", "renovation",
    "wellness retreats", "tornadoes", "hospital", "political ideologies", "lifestyle",
    "pets", "leadership", "social norms", "online learning", "wildlife",
    "career development", "climate change", "international cuisine", "finance",
    "cold", "meat", "theater", "hobbies", "drinks", "family", "lyrics", "gardening",
    "snowfall", "fitness apps", "social issues", "exercise", "clothes", "chatbot",
    "fitness equipment", "model", "home decor", "traditions", "food blogs", "rhythm",
    "instagram", "parenting tips", "plant", "movie genres", "game streaming",
    "vacation stories", "sports events", "paper crafts", "sustainable agriculture",
    "home repairs", "stars", "cryptocurrency", "embroidery", "videogames",
    "self-discipline", "space", "beauty trends", "cinema", "interview",
    "existentialism", "biology", "conflict resolution", "scientific research",
    "sculpture", "professor", "synopsis", "self-care", "science news", "translation",
    "economic theories", "disease", "adventure", "light", "classic literature",
    "flower gardening", "fashion trends", "nutritional supplements", "workout routines",
    "hot", "subject", "weather", "jewelry making", "astronomy", "gardening tips",
    "adventure sports", "crafting", "eco-friendly practices", "collecting",
    "art history", "memorable trips", "fake news", "boat", "productivity", "dress",
    "bike", "lens", "inequality", "environmental conservation", "digital privacy",
    "trending", "society", "scrapbooking", "gadgets", "modern architecture", "botany",
    "celebrities", "accessories", "government policies", "math", "recipes", "quilting",
    "home", "bitcoins", "robots", "design", "car reviews", "beauty products",
    "mobile games", "international relations", "sustainability", "home entertainment",
    "entrepreneurship", "nature documentaries", "mindfulness retreats", "movies",
    "camera", "research", "benefits", "stress management", "elections",
    "autonomous vehicles", "chef recommendations", "cultural heritage", "languages",
    "plane", "parenting challenges", "farming techniques", "cats", "corporate culture",
    "sell", "exoplanets", "console", "budgeting", "music genre", "personality traits",
    "mindfulness", "environment", "press", "travel experiences", "player",
    "game development", "goal setting", "travel tips", "poll", "plant genetics",
    "long-distance relationships", "unsolved mysteries", "language apps", "crafts",
    "fish", "augmented reality", "wearable tech", "self-improvement", "painting",
    "nintendo", "esports", "black holes", "vegetable gardening", "saving",
    "investment strategies", "crypto trading", "behavior", "human actions",
    "human reactions", "human conscious", "human unconscious", "mental processes",
    "thoughts", "feelings", "motives", "medicine",
]


# ---------------------------------------------------------------------------
# Topic / emotion combinations
# ---------------------------------------------------------------------------

def build_topics_emotions(
    n_total_prompts: int = 500,
    n_dialogues_prompt: int = 2,
    turns_per_dialogue: int = 4,
    seed: int = 42,
) -> list[list]:
    """Pair each topic with a list of (human_emotion, chatbot_emotion) tuples."""
    emotions = list(EMOTIONS)
    combinations_sort = list(itertools.combinations_with_replacement(emotions, 2))
    emotions.reverse()
    combinations_reverse = list(itertools.combinations(emotions, 2))
    pair_pool = combinations_sort + combinations_reverse

    n_pair_emotions_prompt = n_dialogues_prompt * turns_per_dialogue
    n_pair_emotions_total = n_total_prompts * n_pair_emotions_prompt
    long_pool = pair_pool * ceil(n_pair_emotions_total / len(pair_pool))

    random.seed(seed)
    random.shuffle(long_pool)

    chunks = [
        long_pool[i : i + n_pair_emotions_prompt]
        for i in range(0, len(long_pool), n_pair_emotions_prompt)
    ]

    return [[topic, emotion] for topic, emotion in zip(TOPICS, chunks)]


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
# Dialogue formatting (used by 3-train_test_rm_prompt_dataset_json)
# ---------------------------------------------------------------------------

DIALOGUE_SYSTEM_TEMPLATE = """You are an expert at creating dialogues.

Dialogue and emotional structure:
Human: ({he1}) PROMPT.
Chatbot: ({he1}) RESPONSE_1. ({ce1}) RESPONSE_2. (NEUTRAL) RESPONSE_3.
Human: ({he2}) PROMPT.
Chatbot: ({he2}) RESPONSE_1. ({ce2}) RESPONSE_2. (NEUTRAL) RESPONSE_3.
Human: ({he3}) PROMPT.
Chatbot: ({he3}) RESPONSE_1. ({ce3}) RESPONSE_2. (NEUTRAL) RESPONSE_3.
Human: ({he4}) PROMPT.
Chatbot: ({he4}) RESPONSE_1. ({ce4}) RESPONSE_2. (NEUTRAL) RESPONSE_3.

Dialogue rules:
The response must be open-domain curated. The response should be coherent, empathetic, engaging and proactive.
The chatbot RESPONSE is composed of 3 different sentences (RESPONSE_1, RESPONSE_2 and RESPONSE_3), separated by a period.
Between RESPONSE_1, RESPONSE_2 and RESPONSE_3 should be a max length of 20-25 words.
RESPONSE_3 must be open-ended to follow-up the conversation, so the Human is encouraged to answer with a full long sentence. Avoid yes/no questions.

Emotional response rules:
RESPONSE_1 must contain a {he4} tone.
RESPONSE_2 must contain a {ce4} tone.
RESPONSE_3 must contain a NEUTRAL tone.

Answer in a single turn to Human. Follow exactly the emotional structure and the emotional and dialogue rules."""



def build_dialogue_record(df_dialogue, dialogue_id: str) -> dict:
    """Convert four turns of a dialogue DataFrame into a single training-record dict."""
    p = df_dialogue["PROMPT"].tolist()
    c = df_dialogue["COMPLETION"].tolist()
    he = df_dialogue["EMOTION_PROMPT"].tolist()
    ce = df_dialogue["EMOTION_RESPONSE_2"].tolist()

    system = DIALOGUE_SYSTEM_TEMPLATE.format(
        he1=he[0], ce1=ce[0], he2=he[1], ce2=ce[1],
        he3=he[2], ce3=ce[2], he4=he[3], ce4=ce[3],
    )
    history = [
        [f"({he[0]}) {p[0]}", c[0]],
        [f"({he[1]}) {p[1]}", c[1]],
        [f"({he[2]}) {p[2]}", c[2]],
    ]
    instruction = f"({he[3]}) {p[3]}"
    return {
        "system": system,
        "history": history,
        "instruction": instruction,
        "input": "",
        "output": c[3],
        "dialogue_id": dialogue_id,
        "set": DATASET_SET,
    }
