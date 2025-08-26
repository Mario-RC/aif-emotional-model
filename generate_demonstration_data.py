import openai
import tiktoken
import pandas as pd
import time
from tqdm import tqdm
import itertools, random

import json
with open("config_chatgpt.json") as json_data_file:
    cfg = json.load(json_data_file)

import logging
logging.basicConfig(level=logging.INFO)
import warnings; warnings.simplefilter('ignore')

dash_line = '-'.join('' for x in range(100))


##########################################
############# PROMPT TEMPLATE ############
##########################################

system = f"""You are an expert at creating emotional dialogues."""

instructions = f"""

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

example = f"""Dialogue example:
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

prompt_template = f"""Follow exactly all dialogues and emotional rules, emotional definitions, emotional structure and the dialogue example. RESPONSE_3 must be a sentence."""


##########################################
########### TOPICS AND EMOTIONS ##########
##########################################

def get_topics_emotions():
    
    topics = ['books', 'author', 'bestsellers', 'education technology', 'online learning', 'photography', 'camera', 'lens', 'light', 'optics', 'zoom', 'portrait', 'animals', 'cats', 'dogs', 'pets', 'birds', 'reptiles', 'wildlife conservation', 'art', 'ballet', 'cinema', 'museum', 'painting', 'theater', 'sculpture', 'street art', 'art history', 'astronomy', 'galaxy', 'planets', 'stars', 'universe', 'black holes', 'constellations', 'space exploration', 'education', 'mark', 'professor', 'school', 'subject', 'university', 'student loans', 'academic research', 'family', 'parents', 'friends', 'relatives', 'marriage', 'children', 'siblings', 'family traditions', 'parenting', 'fashion', 'catwalk', 'clothes', 'design', 'dress', 'footwear', 'jewel', 'model', 'accessories', 'fashion trends', 'haute couture', 'finance', 'benefits', 'bitcoins', 'buy', 'finances', 'investment', 'sell', 'stock market', 'taxes', 'credit cards', 'budgeting', 'retirement planning', 'food', 'drinks', 'fish', 'healthy food', 'meal', 'meat', 'vegetables', 'dessert', 'cooking techniques', 'international cuisine', 'food festivals', 'vegan', 'movies', 'actor', 'director', 'movie genres', 'movies plot', 'synopsis', 'film festivals', 'movie awards', 'film analysis', 'music', 'band', 'dance', 'music genre', 'lyrics', 'rhythm', 'singer', 'song', 'concerts', 'news', 'exclusive', 'fake news', 'interview', 'trending', 'headlines', 'journalism', 'press', 'nutrition', 'allergies', 'diabetes', 'diet', 'obesity', 'nutritional supplements', 'meal planning', 'politics', 'elections', 'poll', 'vote', 'political ideologies', 'government policies', 'science', 'biology', 'math', 'nature', 'physics', 'robots', 'space', 'chemistry', 'scientific discoveries', 'social media', 'facebook', 'instagram', 'twitter', 'snapchat', 'linkedin', 'tiktok', 'society', 'culture', 'holiday', 'party', 'relations', 'wedding', 'multiculturalism', 'social norms', 'community engagement', 'sports', 'baseball', 'basketball', 'coach', 'exercise', 'football', 'player', 'soccer', 'tennis', 'gymnastics', 'golf', 'sportsmanship', 'vehicle', 'bike', 'boat', 'car', 'failure', 'fuel', 'parts', 'plane', 'public transport', 'vehicle speed', 'electric cars', 'autonomous vehicles', 'transportation', 'videogames', 'arcade', 'computer', 'console', 'nintendo', 'play station', 'xbox', 'vr gaming', 'esports', 'weather', 'cloudy', 'cold', 'hot', 'raining', 'sunny', 'snowfall', 'tornadoes', 'climate patterns', 'healthcare', 'disease', 'hospital', 'loneliness', 'mental health', 'nurse', 'physician', 'therapy', 'primary care', 'healthcare disparities', 'research', 'AI', 'experiment', 'investigation', 'survey', 'qualitative research', 'fieldwork', 'academic journals', 'botany', 'flowers', 'fruit', 'plant', 'trees', 'botanical gardens', 'horticulture', 'plant genetics', 'travel', 'destinations', 'adventure', 'backpacking', 'cultural experiences', 'travel tips', 'world landmarks', 'technology', 'gadgets', 'virtual reality', 'wearable tech', 'tech innovations', 'environment', 'climate change', 'sustainability', 'renewable energy', 'eco-friendly practices', 'conservation efforts', 'psychology', 'emotions', 'therapy techniques', 'mental disorders', 'personality traits', 'hobbies', 'crafts', 'knitting', 'woodworking', 'scrapbooking', 'gardening', 'home', 'interior design', 'architecture', 'home improvement', 'Feng Shui', 'smart homes', 'organization', 'relationships', 'dating', 'communication', 'long-distance relationships', 'conflict resolution', 'pop culture', 'celebrities', 'gossip', 'music charts', 'trends in fashion and entertainment', 'viral memes', 'science fiction', 'fantasy', 'time travel', 'alternate realities', 'technology trends', 'artificial intelligence', 'augmented reality', '5G', 'digital privacy', 'travel experiences', 'vacation stories', 'memorable trips', 'local cuisine', 'lifestyle', 'self-care', 'wellness', 'mindfulness', 'meditation', 'stress management', 'healthy habits', 'gaming', 'board games', 'mobile games', 'game development', 'game design', 'game streaming', 'cooking', 'recipes', 'culinary techniques', 'cooking competitions', 'food blogs', 'chef recommendations', 'current events', 'global issues', 'humanitarian efforts', 'international relations', 'history', 'ancient civilizations', 'historical events', 'unsolved mysteries', 'historical figures', 'environmental conservation', 'endangered species', 'green initiatives', 'sustainable living', 'home entertainment', 'streaming services', 'binge-watching', 'DIY projects', 'home decor', 'crafting', 'upcycling', 'interior design ideas']
    random.shuffle(topics)
    
    [('FEAR', 'NEUTRAL'), ('FEAR', 'HAPPINESS'), ('HAPPINESS', 'FEAR'), ('SURPRISE', 'HAPPINESS'), ('DISGUST', 'SURPRISE'), ('DISGUST', 'DISGUST'), ('HAPPINESS', 'HAPPINESS'), ('FEAR', 'HAPPINESS'), ('NEUTRAL', 'NEUTRAL'), ('SURPRISE', 'DISGUST'), ('FEAR', 'NEUTRAL'), ('DISGUST', 'FEAR')]

    emotions = ['ANGER', 'DISGUST', 'FEAR', 'HAPPINESS', 'SADNESS', 'SURPRISE', 'NEUTRAL']
    combinations_sort = list(itertools.combinations_with_replacement(emotions, 2))
    emotions.reverse()
    combinations_reverse = list(itertools.combinations(emotions, 2))
    combinations = combinations_sort + combinations_reverse

    combinations_long = combinations*600
    random.shuffle(combinations_long)

    N = 12
    combinationsList = [combinations_long[n:n+N] for n in range(0, len(combinations_long), N)]

    topics_emotions_dialogues = [[topic, emotion] for topic, emotion in zip(topics, combinationsList)]
    
    return topics_emotions_dialogues


##########################################
############### LOAD MODEL ###############
##########################################

params = {
    "MODEL" : cfg["MODEL"],
    "OPENAI_API_BASE" : cfg["OPENAI_API_BASE"],
    "OPENAI_API_TYPE" : cfg["OPENAI_API_TYPE"],
    "OPENAI_API_VERSION" : cfg["OPENAI_API_VERSION"],
    "OPENAI_API_KEY" : cfg["OPENAI_API_KEY"]
}

openai.api_key = params['OPENAI_API_KEY']
openai.api_base =  params['OPENAI_API_BASE']
openai.api_type = params['OPENAI_API_TYPE']
openai.api_version = params['OPENAI_API_VERSION']


##########################################
################## PROMPT ################
##########################################

def make_prompt(topic, emotion):
    """
    Create the prompt for the LLM
    """
    
    prompt = f"""Provide 3 new different dialogues where the topic of the conversations is {topic}. """

    dialogue_emotions = f"""

Dialogue 1 emotions:
Turn 1 Human 1: ({emotion[0][0]}) PROMPT.
Turn 1 Human 2: ({emotion[0][0]}) RESPONSE_1.
({emotion[0][1]}) Let's think step by step. Step 1: The response must contain an emotional tone of {emotion[0][1]}.
Step 2: Human 2 should express 
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.
Turn 2 Human 1: ({emotion[1][0]}) PROMPT.
Turn 2 Human 2: ({emotion[1][0]}) RESPONSE_1.
({emotion[1][1]}) Let's think step by step. Step 1: The response must contain an emotional tone of {emotion[1][1]}.
Step 2: Human 2 should express 
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.
Turn 3 Human 1: ({emotion[2][0]}) PROMPT.
Turn 3 Human 2: ({emotion[2][0]}) RESPONSE_1.
({emotion[2][1]}) Let's think step by step. Step 1: The response must contain an emotional tone of {emotion[2][1]}.
Step 2: Human 2 should express 
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.
Turn 4 Human 1: ({emotion[3][0]}) PROMPT.
Turn 4 Human 2: ({emotion[3][0]}) RESPONSE_1.
({emotion[3][1]}) Let's think step by step. Step 1: The response must contain an emotional tone of {emotion[3][1]}.
Step 2: Human 2 should express 
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.

Dialogue 2 emotions:
Turn 1 Human 1: ({emotion[4][0]}) PROMPT.
Turn 1 Human 2: ({emotion[4][0]}) RESPONSE_1.
({emotion[4][1]}) Let's think step by step. Step 1: The response must contain an emotional tone of {emotion[4][1]}.
Step 2: Human 2 should express 
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.
Turn 2 Human 1: ({emotion[5][0]}) PROMPT.
Turn 2 Human 2: ({emotion[5][0]}) RESPONSE_1.
({emotion[5][1]}) Let's think step by step. Step 1: The response must contain an emotional tone of {emotion[5][1]}.
Step 2: Human 2 should express 
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.
Turn 3 Human 1: ({emotion[6][0]}) PROMPT.
Turn 3 Human 2: ({emotion[6][0]}) RESPONSE_1.
({emotion[6][1]}) Let's think step by step. Step 1: The response must contain an emotional tone of {emotion[6][1]}.
Step 2: Human 2 should express 
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.
Turn 4 Human 1: ({emotion[7][0]}) PROMPT.
Turn 4 Human 2: ({emotion[7][0]}) RESPONSE_1.
({emotion[7][1]}) Let's think step by step. Step 1: The response must contain an emotional tone of {emotion[7][1]}.
Step 2: Human 2 should express 
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.

Dialogue 3 emotions:
Turn 1 Human 1: ({emotion[8][0]}) PROMPT.
Turn 1 Human 2: ({emotion[8][0]}) RESPONSE_1.
({emotion[8][1]}) Let's think step by step. Step 1: The response must contain an emotional tone of {emotion[8][1]}.
Step 2: Human 2 should express 
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.
Turn 2 Human 1: ({emotion[9][0]}) PROMPT.
Turn 2 Human 2: ({emotion[9][0]}) RESPONSE_1.
({emotion[9][1]}) Let's think step by step. Step 1: The response must contain an emotional tone of {emotion[9][1]}.
Step 2: Human 2 should express 
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.
Turn 3 Human 1: ({emotion[10][0]}) PROMPT.
Turn 3 Human 2: ({emotion[10][0]}) RESPONSE_1.
({emotion[10][1]}) Let's think step by step. Step 1: The response must contain an emotional tone of {emotion[10][1]}.
Step 2: Human 2 should express 
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3.
Turn 4 Human 1: ({emotion[11][0]}) PROMPT.
Turn 4 Human 2: ({emotion[11][0]}) RESPONSE_1.
({emotion[11][1]}) Let's think step by step. Step 1: The response must contain an emotional tone of {emotion[11][1]}.
Step 2: Human 2 should express 
Step 3: RESPONSE_2.
(NEUTRAL) RESPONSE_3."""
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instructions},
        {"role": "assistant", "content": example},
        {"role": "user", "content": prompt_template+prompt+dialogue_emotions}
    ]

    print('\n' + system + '\n' + instructions + '\n' + example + '\n' + prompt_template + prompt + dialogue_emotions)

    return messages


##########################################
############### COMPLETION ###############
##########################################

def handle_message(topic, emotion):
    """
    Get completion
    """

    time.sleep(1)
    prompt = make_prompt(topic, emotion)
    response = openai.ChatCompletion.create(engine="ChatGPT",messages=prompt)
    completion = response.choices[0]['message']['content']
    
    return prompt, completion


##########################################
################## MAIN ##################
##########################################

if __name__=="__main__":

    did, prompts, completions, topic_list, emotions_list = [], [], [], [], []
    chkpt = 0

    topics_emotions_dialogues = get_topics_emotions()

    for idx, topic_emotions in tqdm(enumerate(topics_emotions_dialogues, start=1)):

        topic = topic_emotions[0]
        emotions = topic_emotions[1]
        prompt, completion = handle_message(topic, emotions)
        
        did.append('CHATGPT-' + str(idx-1).zfill(4))
        prompts.append(prompt)
        completions.append(completion)
        topic_list.append(topic)
        emotions_list.append(emotions)

        if idx % 30 == 0:
            print('iter: ' + str(idx))
            zipped_summaries = list(zip(did, prompts, completions, topic_list, emotions_list))
            df_checkpoint = pd.DataFrame(zipped_summaries, columns = ['DID', 'PROMPT', 'COMPLETION', 'TOPIC', 'EMOTIONS'])
            df_checkpoint.to_csv('demonstration_data/demonstration_data/demonstation_data_checkpoint_ ' + str(chkpt) + '.csv', index=False)
            chkpt += 1

zipped_summaries = list(zip(did, prompts, completions, topic_list, emotions_list))
df = pd.DataFrame(zipped_summaries, columns = ['DID', 'PROMPT', 'COMPLETION', 'TOPIC', 'EMOTIONS'])
print(df)

df.to_csv('demonstration_data/demonstration_data/demonstration_data.csv', index=False)
