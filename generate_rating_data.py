import time
import pandas as pd
import json
from tqdm import tqdm
from openai import AzureOpenAI
import anthropic
import google.generativeai as genai
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage

import logging
logging.basicConfig(level=logging.INFO)
import warnings; warnings.simplefilter('ignore')

config_file = "config_llm.json"
with open(config_file) as json_data_file:
    cfg = json.load(json_data_file)

# LLM = "GPT-4O"
# LLM = "GEMINI-1.5-PRO"
# LLM = "CLAUDE-3.5-SONNET"
# LLM = "LLAMA-3.1-405B"

cfg = cfg[LLM]
print(cfg)


#########################################
############### LOAD DATA ###############
#########################################
data_filename = 'data/aif_data.json'
with open(data_filename, 'r', encoding='utf-8') as f:
    aif_data = json.loads(f.read())
f.close()


##########################################
############### LOAD MODEL ###############
##########################################

if (LLM == "CHATGPT") or (LLM == "GPT-4") or (LLM == "GPT-4-TURBO") or (LLM == "GPT-4O"):
    client = AzureOpenAI(
    azure_endpoint =  cfg["AZURE_OPENAI_ENDPOINT"],
    api_key = cfg["AZURE_OPENAI_API_KEY"],
    api_version = cfg["OPENAI_API_VERSION"],
    )
elif (LLM == "CLAUDE-3-OPUS") or (LLM == "CLAUDE-3.5-SONNET"):
    client = anthropic.Anthropic(
        api_key=cfg["ANTHROPIC_API_KEY"],
    )
elif (LLM == "GEMINI-1.0-PRO") or (LLM == "GEMINI-1.5-FLASH") or (LLM == "GEMINI-1.5-PRO"):
    genai.configure(api_key=cfg["GEMINI_API_KEY"])
elif (LLM == "LLAMA-3.1-405B"):
    client = ChatCompletionsClient(
        endpoint=cfg["ENDPOINT"],
        credential=AzureKeyCredential(cfg["AZURE_INFERENCE_CREDENTIAL"])
    )


##########################################
################ EMOTIONS ################
##########################################

def split_emo_utt(prompt, target):

    emo_ini, emo_end = '(', ')'
    res_ini = [i for i in range(len(prompt)) if prompt.startswith(emo_ini, i)]
    res_end = [i for i in range(len(prompt)) if prompt.startswith(emo_end, i)]

    emo1 = prompt[res_ini[0]+1:res_end[0]].strip()

    emo_ini, emo_end = '(', ')'
    res_ini = [i for i in range(len(target)) if target.startswith(emo_ini, i)]
    res_end = [i for i in range(len(target)) if target.startswith(emo_end, i)]

    emo2 = target[res_ini[1]+1:res_end[1]].strip()

    return emo1, emo2


def get_emotions_table(human_emo, chatbot_emo):
    
    empathy_level, emotion_level, question_level = 0, 0, 0

    emotions = ['ANGER', 'FEAR', 'SADNESS', 'DISGUST', 'HAPPINESS', 'SURPRISE', 'NEUTRAL']

    if human_emo == emotions[0]:
        empathy_level = 5
        emotion_level = [1, 1, 2, 1, 2, 4, 5]
        question_level = 5
    elif human_emo == emotions[1]:
        empathy_level = 5
        emotion_level = [1, 1, 2, 2, 2, 4, 5]
        question_level = 5
    elif human_emo == emotions[2]:
        empathy_level = 5
        emotion_level = [1, 1, 4, 1, 1, 4, 5]
        question_level = 5
    elif human_emo == emotions[3]:
        empathy_level = 4
        emotion_level = [2, 2, 2, 4, 5, 4, 1]
        question_level = 4
    elif human_emo == emotions[4]:
        empathy_level = 3
        emotion_level = [5, 5, 5, 4, 5, 5, 5]
        question_level = 3
    elif human_emo == emotions[5]:
        empathy_level = 4
        emotion_level = [4, 4, 4, 4, 4, 4, 4]
        question_level = 4
    elif human_emo == emotions[6]:
        empathy_level = 3
        emotion_level = [5, 5, 5, 4, 5, 5, 5]
        question_level = 5

    if chatbot_emo == emotions[0]:
        emotion_level = emotion_level[0]
    elif chatbot_emo == emotions[1]:
        emotion_level = emotion_level[1]
    elif chatbot_emo == emotions[2]:
        emotion_level = emotion_level[2]
    elif chatbot_emo == emotions[3]:
        emotion_level = emotion_level[3]
    elif chatbot_emo == emotions[4]:
        emotion_level = emotion_level[4]
    elif chatbot_emo == emotions[5]:
        emotion_level = emotion_level[5]
    elif chatbot_emo == emotions[6]:
        emotion_level = emotion_level[6]

    return empathy_level, emotion_level, question_level


def rating_rules(empathy_level, emotion_level, question_level, human_emo, chatbot_emo):
    
    # Different rating rules for each emotion level could be applied if needed
    # if empathy_level == 1:
    #     empathy_rule = f"""- RESPONSE_1 must not be empathetic at all to the human UTTERANCE. The less empathetic it expresses, the higher the score, and vice versa."""
    # elif empathy_level == 2:
    #     empathy_rule = f"""- RESPONSE_1 must be slightly empathetic to the human UTTERANCE. The less slightly empathetic it expresses, the higher the score, and vice versa."""
    # elif empathy_level == 3:
    #     empathy_rule = f"""- RESPONSE_1 could be empathetic to the human UTTERANCE."""
    # elif empathy_level == 4:
    #     empathy_rule = f"""- RESPONSE_1 must be empathetic to the human UTTERANCE."""
    # elif empathy_level == 5:
    #     empathy_rule = f"""- RESPONSE_1 must be highly empathetic to the human UTTERANCE."""
    
    # if emotion_level == 1:
    #     emotion_rule = f"""- RESPONSE_2 must not express a {chatbot_emo} emotional tone. The less emotion it expresses, the higher the score, and vice versa."""
    # elif emotion_level == 2:
    #     emotion_rule = f"""- RESPONSE_2 must express a slightly {chatbot_emo} emotional tone."""
    # elif emotion_level == 3:
    #     emotion_rule = f"""- RESPONSE_2 could express some {chatbot_emo} emotional tone."""
    # elif emotion_level == 4:
    #     emotion_rule = f"""- RESPONSE_2 must express some {chatbot_emo} emotional tone."""
    # elif emotion_level == 5:
    #     emotion_rule = f"""- RESPONSE_2 must express a very strong {chatbot_emo} emotional tone."""
    
    # if question_level == 1:
    #     question_rule = f"""- RESPONSE_3 must not be a follow-up question that encourages the user for further conversation. Closed-ended questions, statements or yes/no questions must have a very high score."""
    # elif question_level == 2:
    #     question_rule = f"""- RESPONSE_3 must be a slightly follow-up question to encourage the user to continue the conversation. Closed-ended questions, statements or yes/no questions must have a high score."""
    # elif question_level == 3:
    #     question_rule = f"""- RESPONSE_3 could end with a good follow-up question that encourages the user for further conversation."""
    # elif question_level == 4:
    #     question_rule = f"""- RESPONSE_3 must be a good follow-up question that encourages the user for further conversation. Closed-ended questions, statements or yes/no questions must have a low score."""
    # elif question_level == 5:
    #     question_rule = f"""- RESPONSE_3 must be a prominent follow-up question that encourages the user for further conversation. Closed-ended questions, statements or yes/no questions must have a very low score."""

    empathy_rule = f"""- RESPONSE_1 must be very empathetic to human UTTERANCE that expresses a {human_emo} emotional tone. The more empathy it expresses, the higher the score, and vice versa."""
    emotion_rule = f"""- RESPONSE_2 must express a very strong {chatbot_emo} emotional tone. The more emotion it expresses, the higher the score, and vice versa."""
    question_rule = f"""- RESPONSE_3 must be a prominent follow-up question that encourages the user for further conversation. The better follow-up question is expressed, the higher the score, and vice versa. Closed-ended questions, statements or yes/no questions must have a very low score."""

    return empathy_rule, emotion_rule, question_rule

##########################################
################## PROMPT ################
##########################################

def make_prompt(entry, empathy_rule, emotion_rule, question_rule):
    """
    Create the prompt for the LLM
    """

    system = f"""You are an expert at scoring responses that contain empathy, emotion and follow-up questions."""

    instructions = f"""This is a conversation on a specific topic and with a certain emotional tone in each turn. At the end of the conversation, 10 different answers have been created for the last chatbot turn.

Dialogue structure:
All human turns convey a specific emotion following structure: (HUMAN_EMOTION) UTTERANCE.
All chatbot turns are composed by 3 different sentences, separated by a period. The first sentence is empathetic regarding the previous human utterance, the second sentence follows a specific internal chatbot emotion and the third sentences is a follow-up question. This is the structure: (EMPATHY) RESPONSE_1, (CHATBOT_EMOTION) RESPONSE_2, (QUESTION) RESPONSE_3."""

    rating_rules = f"""Rating rules:
{empathy_rule}
{emotion_rule}
{question_rule}"""

    predicts = f"""Human-Chatbot conversation:
Human: {entry['history'][0][0]}
Chatbot: {entry['history'][0][1]}
Human: {entry['history'][1][0]}
Chatbot: {entry['history'][1][1]}
Human: {entry['history'][2][0]}
Chatbot: {entry['history'][2][1]}
Human: {entry['prompt']}

There are 10 different possible chatbot answers, from A1 to A10.
A1 - {entry["target"]}
A2 - {entry["predict_1"]}
A3 - {entry["predict_2"]}
A4 - {entry["predict_3"]}
A5 - {entry["predict_4"]}
A6 - {entry["predict_5"]}
A7 - {entry["predict_6"]}
A8 - {entry["predict_7"]}
A9 - {entry["predict_8"]}
A10 - {entry["predict_9"]}

Base the scoring according to the three following rating rules, scores the adherence of each of the sentences to its respective rating rule, that is EMPATHY in RESPONSE_1, CHATBOT_EMOTION in RESPONSE_2 and QUESTION in RESPONSE_3 for each answer. Score from 1 to 9, where 9 is better than 1. Consider that the more closely the sentence fits its rating rule, the higher its score will be. Just classify each of the answers according to the following structure. No additional explanations or justifications are needed.

A1: Empathy: ,Chatbot Emotion: ,Question: 
A2: Empathy: ,Chatbot Emotion: ,Question: 
A3: Empathy: ,Chatbot Emotion: ,Question: 
A4: Empathy: ,Chatbot Emotion: ,Question: 
A5: Empathy: ,Chatbot Emotion: ,Question: 
A6: Empathy: ,Chatbot Emotion: ,Question: 
A7: Empathy: ,Chatbot Emotion: ,Question: 
A8: Empathy: ,Chatbot Emotion: ,Question: 
A9: Empathy: ,Chatbot Emotion: ,Question: 
A10: Empathy: ,Chatbot Emotion: ,Question: """

    return [system, instructions, rating_rules, predicts]


##########################################
############### COMPLETION ###############
##########################################

def handle_message(prompt):
    """
    Get completion
    """
    system = prompt[0]
    instructions = prompt[1]
    rating_rules = prompt[2]
    predicts = prompt[3]

    # Generate completions
    if LLM == "GPT-4O":
        time.sleep(5)
        response = client.chat.completions.create(
            model=cfg["MODEL"],
            max_tokens=1000,
            # temperature=0,
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": instructions},
                {"role": "assistant", "content": rating_rules},
                {"role": "user", "content": predicts}
            ]
        )
        completion = response.choices[0].message.content

    elif LLM == "CLAUDE-3.5-SONNET":
        time.sleep(5)
        response = client.messages.create(
            model=cfg["MODEL"],
            max_tokens=1000,
            # temperature=0,
            system=system,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": instructions}]},
                {"role": "assistant", "content": [{"type": "text", "text": rating_rules}]},
                {"role": "user", "content": [{"type": "text", "text": predicts}]}
            ]
        )
        completion = response.content[0].text

    elif LLM == "GEMINI-1.5-PRO":
        time.sleep(5)
        prompt = prompt[0]  + '\n\n' + prompt[1]  + '\n\n' + prompt[2]  + '\n\n' + prompt[3]
        model = genai.GenerativeModel(cfg["MODEL"])
        completion = model.generate_content(
            prompt,
            generation_config = genai.GenerationConfig(
                max_output_tokens=1000,
                # temperature=0.1,
            )
        )
        completion = completion.candidates[0].content.parts[0].text

    elif LLM == "LLAMA-3.1-405B":
        time.sleep(5)
        prompt = prompt[1]  + '\n\n' + prompt[2]  + '\n\n' + prompt[3]
        response = client.complete(
            max_tokens=1000,
            # temperature=0.8,
            # top_p=0.1,
            # presence_penalty=0
            messages=[
                SystemMessage(content=system),
                UserMessage(content=prompt),
            ]
        )
        completion = response.choices[0].message.content

    return completion


##########################################
################## MAIN ##################
##########################################

if __name__=="__main__":

    did, prompts, completions, emotions_list, expression_level = [], [], [], [], []
    chkpt = 0
    
    for idx, entry in tqdm(enumerate(aif_data, start=1)):

        did.append(entry['did'])

        human_emo, chatbot_emo = split_emo_utt(entry['prompt'], entry['target'])
        emotions_list.append([human_emo, chatbot_emo])
        
        empathy_level, emotion_level, question_level = get_emotions_table(human_emo, chatbot_emo)
        expression_level.append([empathy_level, emotion_level, question_level])
        
        empathy_rule, emotion_rule, question_rule = rating_rules(empathy_level, emotion_level, question_level, human_emo, chatbot_emo)

        prompt = make_prompt(entry, empathy_rule, emotion_rule, question_rule)
        prompt_message = prompt[0]  + '\n\n' + prompt[1]  + '\n\n' + prompt[2]  + '\n\n' + prompt[3]
        prompts.append(prompt_message)
        
        completion = handle_message(prompt)
        completions.append([completion])

        if idx % 10 == 0:
            zipped_summaries = list(zip(did, prompts, completions, emotions_list, expression_level))
            df_checkpoint = pd.DataFrame(zipped_summaries, columns = ['DID', 'PROMPT', 'COMPLETION', 'EMOTIONS', 'EXPRESSION_LEVEL'])
            df_checkpoint.to_csv('data/' + LLM + '/records/aif_data_rate_' + LLM + '_checkpoint_' + str(chkpt) + '.csv', index=False)
            chkpt += 1


zipped_summaries = list(zip(did, prompts, completions, emotions_list, expression_level))
df = pd.DataFrame(zipped_summaries, columns = ['DID', 'PROMPT', 'COMPLETION', 'EMOTIONS', 'EXPRESSION_LEVEL'])

df.to_csv('data/' + LLM + '/aif_data_rate_' + LLM + '.csv', index=False)
