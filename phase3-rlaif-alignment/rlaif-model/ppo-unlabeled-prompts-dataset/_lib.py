"""Shared utilities for the PPO Unlabeled Prompts Dataset pipeline.

Holds the topic catalogue (1000 topics: the original 500 inherited from the
RM Prompt Dataset plus 500 RLAIF-specific extensions), the emotion list,
dialogue prompt templates and small JSON / dialogue helpers used across the
four numbered scripts.
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
DATASET_SET = "ppo-unlabeled-prompts"

#: First half of the topic list (shared with phase3-rlaif-alignment/reward-model/rm-prompt-dataset).
TOPICS_BASE: list[str] = [
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

#: Second half of the topic list (RLAIF-specific extensions).
TOPICS_RLAIF_EXTRA: list[str] = [
    "genetic engineering", "ocean currents", "quantum computing", "bee conservation",
    "renewable materials", "urban farming", "volcanoes", "forensic science",
    "folk music", "artificial organs", "eco-tourism", "night sky photography",
    "space debris", "medicinal plants", "zero-waste living", "permaculture",
    "astrobiology", "indigenous cultures", "astrophysics", "organic skincare",
    "snowboarding", "robotics in medicine", "vintage fashion", "forest ecosystems",
    "whale migration", "architectural preservation", "underwater archaeology",
    "film soundtracks", "nomadic lifestyles", "forest management", "remote sensing",
    "quantum entanglement", "desert ecosystems", "trampoline workouts",
    "military history", "brain-computer interfaces", "marine biology", "urban wildlife",
    "coral reefs", "ice climbing", "martial arts", "stem education", "spy technology",
    "biodegradable packaging", "nanotechnology", "science fairs", "circus arts",
    "solar flares", "autonomous drones", "space elevators", "ethical hacking",
    "freshwater conservation", "glacier retreats", "paleobotany", "ethical consumerism",
    "small-scale fisheries", "avian migration", "storm chasing", "hybrid cars",
    "whale watching", "energy storage", "space colonization", "horseback riding",
    "bubble tea", "citizen science", "wildfire prevention", "motion capture technology",
    "medieval weaponry", "sustainable packaging", "rainforest biodiversity",
    "food fermentation", "chess strategy", "mythical creatures", "voice acting",
    "water purification", "wildflower identification", "digital detox", "geocaching",
    "cartography", "urban graffiti", "thermal energy", "brain plasticity",
    "steampunk art", "ocean acidification", "bioluminescence", "monsoon patterns",
    "vinyl record collecting", "contemporary dance", "rare gemstones", "climate refugees",
    "literary symbolism", "escape room design", "polaroid photography",
    "mechanical keyboards", "leather crafting", "cybersecurity threats", "kinetic art",
    "ocean exploration", "ethical fashion", "self-driving buses", "bedouin traditions",
    "snow leopard conservation", "celestial navigation", "seed banks", "rural education",
    "e-sports casting", "genealogy", "mountaineering safety", "arctic wildlife",
    "drone racing", "history of tea", "stone carving", "augmented reality glasses",
    "online activism", "blacksmithing", "coastal restoration", "typography design",
    "lighthouses", "polar science", "skin microbiome", "adventure novels",
    "contemporary sculpture", "viking history", "balloon art", "iceberg monitoring",
    "diy solar panels", "traditional calligraphy", "puppetry", "eco-friendly homes",
    "miniature painting", "microbial life", "antarctic expeditions", "yoga retreats",
    "waterfall exploration", "historical costumes", "spearfishing", "instrument making",
    "folklore tales", "ancient pottery", "moss gardening", "water management",
    "rare book restoration", "color psychology", "hot air ballooning",
    "submarine technology", "medieval literature", "telescope maintenance",
    "beach cleanup initiatives", "combat sports", "hiking trails", "enamel art",
    "opera performances", "carnivorous plants", "paper engineering",
    "wastewater recycling", "glassblowing", "snow sculpting", "ethical tourism",
    "bonsai art", "mushroom cultivation", "wind energy", "lunar phases",
    "archaeological digs", "textile dyeing", "eco-friendly transportation",
    "wildlife tracking", "historical reenactments", "custom footwear", "parkour",
    "honey production", "fantasy novels", "biodiversity mapping", "aerospace engineering",
    "seismology", "skateboarding culture", "desert survival", "aquaponics",
    "climate modeling", "biodynamic farming", "urban green spaces",
    "underwater photography", "cultural festivals", "night markets", "map making",
    "virtual museums", "ceramic restoration", "natural building materials",
    "rocket propulsion", "medicinal fungi", "river ecosystems", "eco-fashion brands",
    "snowshoeing", "acoustic ecology", "forest therapy", "outdoor cooking",
    "folktale analysis", "portable shelters", "meteorite hunting", "salt flats",
    "ethical ai", "butterfly migration", "sustainable cities", "eco-villages",
    "urban legends", "heritage crafts", "climate communication", "mycology",
    "nature reserves", "indigenous art", "art installations", "fitness wearables",
    "desert blooms", "planetary geology", "repurposed materials", "traditional weaving",
    "habitat restoration", "geothermal energy", "treehouse building", "free diving",
    "natural soundscapes", "cave paintings", "led art", "arctic exploration",
    "indigenous languages", "surfboard design", "museum curation", "ethical robotics",
    "polar bear conservation", "fair trade practices", "diy composting", "solar cooking",
    "algae farming", "mushroom foraging", "natural pest control", "pebble mosaics",
    "rewilding projects", "glacier trekking", "conservation photography", "origami",
    "wildlife corridors", "digital archiving", "endemic species", "solar sails",
    "cloud formations", "kitesurfing", "songbirds", "wild mushroom identification",
    "habitat mapping", "freshwater ecosystems", "crow intelligence",
    "acoustic instruments", "paleoclimatology", "starfish biology",
    "water scarcity solutions", "gliding sports", "reforestation techniques",
    "fossil preparation", "permian extinction", "seagrass meadows", "mangrove swamps",
    "mountain biking", "lichen studies", "renewable textiles", "ice age megafauna",
    "edible insects", "urban farming techniques", "climbing walls", "solar architecture",
    "fossil fuels transition", "moon colonies", "seabird conservation",
    "woodblock printing", "lightning phenomena", "tribal jewelry", "cave diving",
    "wool spinning", "whale songs", "beach ecosystems", "eco-friendly festivals",
    "wildflower propagation", "natural stone sculptures", "wilderness survival",
    "glacier ecology", "rainwater harvesting", "ancient writing systems",
    "ice core analysis", "sustainable fisheries", "biodiversity hotspots",
    "desert flora", "sailing navigation", "star charts", "aquatic plant life",
    "snowboarding gear", "ice caves", "traditional fishing techniques",
    "cave ecosystems", "cliff diving", "ethical startups", "storm surge management",
    "volcanic islands", "meteorological instruments", "vintage toys",
    "mangrove restoration", "steam power", "fairy tale origins", "endemic flora",
    "biogas production", "glass recycling", "rocket modeling", "living off-grid",
    "bird sanctuaries", "natural pigments", "fossil sites", "permaculture design",
    "sand dune conservation", "tree climbing", "organic textiles", "beachcombing",
    "environmental storytelling", "fern gardening", "seasonal farming", "ice fishing",
    "polar ice caps", "adaptive clothing", "sustainable tourism models",
    "aurora borealis", "regenerative farming", "salt marshes", "freshwater species",
    "tidal energy", "ecosystem restoration", "biodegradable materials",
    "oceanography instruments", "energy-efficient lighting", "non-violent communication",
    "digital nomadism", "algal blooms", "deep sea vents", "sustainable eating",
    "shell collecting", "birdsong identification", "biotic communities",
    "algae-based products", "genetic sequencing", "ethical investments",
    "living architecture", "tundra ecosystems", "polar expeditions",
    "saltwater aquariums", "zero-emission transportation", "sea turtles",
    "climate policy frameworks", "bison restoration", "cultural adaptation",
    "solar-powered vehicles", "insect hotels", "wilderness trails", "paper restoration",
    "adaptation in nature", "community farming", "bee-friendly gardens",
    "glacier calving", "native bees", "grassland ecosystems", "fog catchers",
    "mobile observatories", "ethical marine practices", "natural dyeing techniques",
    "alkaline soils", "urban air quality", "elephant corridors", "amphibian studies",
    "edible weeds", "mountain shelters", "water desalination", "wildlife sanctuaries",
    "recycled art", "polar sea ice", "eco-friendly packaging", "species adaptation",
    "plant symbiosis", "ethical real estate", "microhydropower", "nutritional ecology",
    "urban food systems", "cold desert ecosystems", "bats as pollinators",
    "rainforest tribes", "wildlife camera traps", "solar rooftops", "bioclimatic design",
    "natural ice rinks", "deep-sea exploration", "arctic food chains", "river deltas",
    "polar research stations", "fossilized forests", "prairie restoration",
    "urban heat islands", "antarctic seals", "food co-ops", "bioluminescent bays",
    "plant nurseries", "ethical education", "nutrient cycling", "perennial crops",
    "coastal dune ecosystems", "desert oases", "recycled construction",
    "biodiversity corridors", "energy-efficient buildings", "seafloor mapping",
    "sunken ships", "fire ecology", "lichen bioindicators", "glacier-fed rivers",
    "natural insulation", "water-saving appliances", "agroforestry", "green roofs",
    "remote ecosystems", "seasonal migration", "dryland farming", "desertification",
    "eco-trails", "ethical volunteering", "beekeeping innovations",
    "wildlife soundtracks", "freshwater wetlands", "weather monitoring", "tidal pools",
    "ethical online communities", "solar villages", "cold water corals",
    "thermal pollution", "ethical journalism", "environmental protest art",
    "wildlife ecology", "marine reserves", "water rights", "renewable polymers",
    "microclimates", "oceanic ridges", "traditional herbal remedies",
    "recycled textiles", "marine geology", "algal bioplastics", "eco-conscious fashion",
    "biotic interactions", "coral bleaching", "solar desalination", "glacier preservation",
    "mangrove ecosystems", "agricultural waste", "ecological calendars", "arctic tundra",
    "river conservation", "fossil fuel impacts", "wildlife population studies",
    "coastal erosion", "energy-efficient appliances", "renewable transport",
    "subarctic ecosystems", "seed saving", "permafrost", "climate-smart farming",
    "eco-gadgets", "crop diversity", "forest floor ecology", "fire-adapted species",
    "eco-markets", "indigenous foods", "botanical illustration", "watershed management",
    "sargassum ecology", "sustainable fiber", "arctic foxes", "windbreaks",
    "wildlife rehabilitation", "bioluminescent fungi", "sea ice dynamics",
    "glacial lakes", "bird banding", "snow crystal formation", "solar-powered gadgets",
    "rural electrification", "glacier hazards", "arctic transportation",
    "energy-positive homes", "wildlife bridges", "solar-heated water",
    "fossil excavations", "cloud ecosystems", "ice sheet dynamics",
    "solar panel recycling", "glacier ice caves", "freshwater habitats",
    "sustainable fishing", "mangrove forests", "indigenous festivals",
]

#: Combined topic list used by 1-generate_ppo_unlabeled_prompts_dataset (1000 entries).
TOPICS: list[str] = TOPICS_BASE + TOPICS_RLAIF_EXTRA


# ---------------------------------------------------------------------------
# Topic / emotion combinations
# ---------------------------------------------------------------------------

def build_topics_emotions(
    n_total_prompts: int = 1000,
    n_dialogues_prompt: int = 1,
    turns_per_dialogue: int = 4,
    seed: int = 42,
) -> list[list]:
    """Pair each topic with a list of (human_emotion, chatbot_emotion) tuples.

    The defaults match the RLAIF setup: 1000 prompts × 1 dialogue × 4 turns.
    """
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
# Dialogue formatting (used by 3-train_test_ppo_unlabeled_prompts_dataset_json)
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


def check_repeated_dialogue_ids(data: list[dict]) -> set[str]:
    """Return the set of ``dialogue_id`` values that occur more than once in ``data``."""
    dialogue_ids = [entry["dialogue_id"] for entry in data]
    return {dialogue_id for dialogue_id in dialogue_ids if dialogue_ids.count(dialogue_id) > 1}
