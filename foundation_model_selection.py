import pandas as pd
import os, re
from tqdm import tqdm
import operator
from functools import reduce
from sklearn.metrics import f1_score, precision_score, recall_score

import json
with open("config.json") as json_data_file:
    cfg = json.load(json_data_file)

import logging
logging.basicConfig(level=logging.INFO)
import warnings; warnings.simplefilter('ignore')

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoModel,
    AutoTokenizer,
    LlamaTokenizer,
    GenerationConfig
)
import torch


prompt_prediction = cfg['args']['dataset']
dataset = cfg['args']['dataset']

model_name = cfg['args']['model']
cuda = cfg['args']['gpu']
debug = cfg['args']['dev']

AUTOMODEL = cfg['automodel']['models']
AUTOMODELFORCASUALLM = cfg['automodelforcasuallm']['models']
LLAMFORCASUALLM = cfg['llamaforcasuallm']['models']
AUTOMODELFORSEQ2SEQLM = cfg['automodelforseq2seqlm']['models']

TRUSTEDREMOTECODE = cfg['trustedremotecode']['models']
REPEATE_PROMPTS = cfg['repeatedprompts']['models']
REPEATE_PROMPTS_LLAMA3 = cfg['repeatedpromptsllama3']['models']
REPEATE_PROMPTS_GEMMA = cfg['repeatedpromptsgemma']['models']
REPEATE_PROMPTS_MISTRAL = cfg['repeatedpromptsmistral']['models']
REPEATE_PROMPTS_CHATGLM3 = cfg['repeatedpromptschatglm3']['models']
REPEATE_PROMPTS_GLM4 = cfg['repeatedpromptsglm4']['models']
REPEATE_PROMPTS_BLOOMZ = cfg['repeatedpromptsbloomz']['models']
REPEATE_PROMPTS_PHI3 = cfg['repeatedpromptsphi3']['models']
REPEATE_PROMPTS_AIRBOROS = cfg['repeatedpromptsairboros']['models']
REPEATE_PROMPTS_ZEPHYR_ALPHA = cfg['repeatedpromptszephyralpha']['models']
REPEATE_PROMPTS_ZEPHYR_BETA = cfg['repeatedpromptszephyrbeta']['models']
REPEATE_PROMPTS_ZEPHYR_GEMMA = cfg['repeatedpromptszephyrgemma']['models']
REPEATE_PROMPTS_INTERNLM = cfg['repeatedpromptsinternlm']['models']

INPUTTEMPLATE1 = cfg['inputtemplate1']['models']
INPUTTEMPLATE2 = cfg['inputtemplate2']['models']
INPUTTEMPLATE3 = cfg['inputtemplate3']['models']
INPUTTEMPLATELLAMA2 = cfg['inputtemplatellama2']['models']
INPUTTEMPLATELLAMA3 = cfg['inputtemplatellama3']['models']
INPUTTEMPLATEGEMMA = cfg['inputtemplategemma']['models']
INPUTTEMPLATEPHI3 = cfg['inputtemplatephi3']['models']
INPUTTEMPLATEMISTRAL = cfg['inputtemplatemistral']['models']
INPUTTEMPLATECHATGLM3 = cfg['inputtemplatechatglm3']['models']
INPUTTEMPLATEGLM4 = cfg['inputtemplateglm4']['models']
INPUTTEMPLATEINTERLM = cfg['inputtemplateinternlm']['models']

dash_line = '-'.join('' for x in range(100))


##########################################
################### DATA #################
##########################################

def download_dataset():

    daily_dialog = load_dataset("daily_dialog")

    uid, did, sid, seg, emotion_list = [], [], [], [], []
    dial_count, turn_count = 0, 0

    dialogs_train = daily_dialog['train']['dialog']
    emotions_train = daily_dialog['train']['emotion']
    dialogs_validation = daily_dialog['validation']['dialog']
    emotions_validation = daily_dialog['validation']['emotion']
    dialogs_test = daily_dialog['test']['dialog']
    emotions_test = daily_dialog['test']['emotion']

    sets = [
        [dialogs_train, emotions_train],
        [dialogs_validation, emotions_validation],
        [dialogs_test, emotions_test]
    ]

    df_sets = []
    for set_ in sets:

        uid, did, sid, seg, emotion_list = [], [], [], [], []
        dial_count, turn_count = 0, 0
        
        for dial, emotion in zip(set_[0], set_[1]):
            turn_count = 0
            for turn, emo in zip(dial, emotion):
                uid.append('DAILYD' + '-' + str(dial_count).zfill(6) + '-' + str(turn_count).zfill(4))
                did.append(uid[-1][:-5])
                sid.append('A' if not turn_count % 2 else 'B')
                seg.append(turn)
                emotion_list.append(emo)
                turn_count += 1
            dial_count += 1

        data = list(zip(uid, did, sid, seg, emotion_list))
        df_sets.append(pd.DataFrame(data, columns = ['UID', 'DID', 'SID', 'SEG', 'EMOTION']))

    data = pd.concat([df_sets[0], df_sets[1], df_sets[2]], ignore_index=True)

    data['EMOTION'] = data['EMOTION'].map({
        0: "neutral",
        1: "anger",
        2: "disgust",
        3: "fear",
        4: "happiness",
        5: "sadness",
        6: "surprise",
    })

    data.to_csv('data/' + dataset + '/' + dataset + '_main.csv', index=False) 

    return data


def create_dialogues():
    """
    Load main dialogues and emotions
    """

    # Check whether the specified path exists or not
    path = 'data/' + dataset + '/' + dataset + '_dialogues.csv'
    isExist = os.path.exists(path)
    if isExist:
        logging.info('DIALOGUES FOUND')
        return pd.read_csv('data/' + dataset + '/' + dataset + '_dialogues.csv')
    
    # Check whether the specified path exists or not
    path = 'data/' + dataset + '/' + dataset + '_main.csv'
    isExist = os.path.exists(path)
    if isExist:
        logging.info('PROCESSED DAILYDIALOG FOUND')
        return pd.read_csv('data/' + dataset + '/' + dataset + '_main.csv')
    
    df_main = download_dataset()

    unique = list(df_main['DID'].unique())
    values = df_main['DID'].value_counts().index.tolist()
    counts = df_main['DID'].value_counts().tolist()

    df_did = pd.DataFrame(unique, columns=['DID_UNIQUE'])

    res = []
    for did_unique in df_did['DID_UNIQUE']:
        idx = values.index(did_unique)
        res.append(counts[idx])
    df_did['COUNT'] = res

    df_did = df_did[df_did['COUNT'] == 4]

    did_unique_count = df_did['DID_UNIQUE'].to_list()
    df_dialogues = df_main[df_main['DID'].isin(did_unique_count)]
    df_dialogues.reset_index(drop=True, inplace=True)
    
    df_dialogues.to_csv('data/' + dataset + '/' + dataset + '_dialogues.csv', index=False) 
    
    return df_dialogues


##########################################
############### LOAD MODEL ###############
##########################################

logging.info('LOADING MODEL: ' + model_name)
model_name_save = model_name.split("/",1)[1]
model_path = 'data/' + dataset + '/' + model_name_save + '/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

trust_remote_code = False
if model_name_save in TRUSTEDREMOTECODE:
    trust_remote_code = True

# Model
if model_name_save in AUTOMODEL:
    model = AutoModel.from_pretrained(
        model_name,
        load_in_4bit=False,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code
        )
elif model_name_save in AUTOMODELFORCASUALLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=False,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code
        )
elif model_name_save in LLAMFORCASUALLM:
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=False,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code
        )
elif model_name_save in AUTOMODELFORSEQ2SEQLM:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        load_in_4bit=False,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code
        )

device = torch.device('cuda:' + str(cuda) if torch.cuda.is_available() else 'cpu')
model.to(device)

# Tokenizer
if model_name_save in LLAMFORCASUALLM:
    tokenizer = LlamaTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=trust_remote_code
        )
elif model_name_save in AUTOMODEL:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=trust_remote_code
        )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        )


##########################################
########## EMOTION COMPLETION ############
##########################################

def get_emotion_completion(
        df_dialogues,
        df_prompt_emotion_completion_1,
        df_prompt_emotion_completion_2,
        df_prompt_emotion_completion_3,
        df_prompt_emotion_completion_4
        ):

    prompt_1 = df_prompt_emotion_completion_1['PROMPT']
    prompt_2 = df_prompt_emotion_completion_2['PROMPT']
    prompt_3 = df_prompt_emotion_completion_3['PROMPT']
    prompt_4 = df_prompt_emotion_completion_4['PROMPT']

    completion_1 = df_prompt_emotion_completion_1['COMPLETION']
    completion_2 = df_prompt_emotion_completion_2['COMPLETION']
    completion_3 = df_prompt_emotion_completion_3['COMPLETION']
    completion_4 = df_prompt_emotion_completion_4['COMPLETION']
    
    prompt = reduce(operator.add, zip(prompt_1, prompt_2, prompt_3, prompt_4))
    completion = reduce(operator.add, zip(completion_1, completion_2, completion_3, completion_4))

    df_dialogues.insert(4, 'PROMPT', prompt)
    df_dialogues.insert(5, 'COMPLETION', completion)
    
    emotions = [
            'anger',
            'disgust',
            'fear',
            'happiness',
            'sadness',
            'surprise',
            'neutral'
            ]
    emotions_options = [
        '[A] Anger',
        '[B] Disgust',
        '[C] Fear',
        '[D] Happiness',
        '[E] Sadness',
        '[F] Surprise',
        '[G] Neutral'
        ]
    
    completion_emotion = []
    completions = df_dialogues['COMPLETION'].to_list()

    for completion in completions:

        try:
            emotion = [e for e in emotions_options if e in completion][0]
            emotion = emotion.split()[-1].lower()
        except:
            try:
                completion = re.sub("['.\[\]!,*)@#%(&$_?.^]", ' ', completion).lower()
                emotion = [e for e in emotions if e in completion][0]
            except:
                emotion = 'none'

        completion_emotion.append(emotion)

    df_dialogues['EMOTION_COMPLETION'] = completion_emotion
    df_dialogues.to_csv('data/' + dataset + '/' + model_name_save + '/' + dataset + '_' + model_name_save + '_emotion_completion_1.csv', index=False)

    return df_dialogues


def check_valid_emotions(df_data):
    """
    Check if the predicted emotions are correct
    """
    
    emotions = list(df_data['EMOTION_COMPLETION'].unique())
    emotions_bool = df_data['EMOTION_COMPLETION'].apply(lambda x: any([k in x for k in emotions]))
    wrong_emotions = [[df_data['EMOTION_COMPLETION'][idx], idx] for idx, boolean in enumerate(emotions_bool) if not boolean]
    print(wrong_emotions)

    return wrong_emotions


def get_metric(y_test, y_pred, turn):

    count = len([i for i, j in zip(y_test, y_pred) if i == j])
    total =len(y_test)
    score = count/total

    if turn == 4:
        print(f'\nAll dataset:\nhits: {count} \ntotal: {total} \nAccuracy: {"%.2f" % (score*100)} %')
    else:
        print(f'\n#turns: {turn} \nhits: {count} \ntotal: {total} \nAccuracy: {"%.2f" % (score*100)} %')
    print("F1: %.4f" % f1_score(y_test, y_pred, average="macro"))
    print("Precision: %.4f" % precision_score(y_test, y_pred, average="macro"))
    print("Recall: %.4f" % recall_score(y_test, y_pred, average="macro"), "\n")


def check_accuracy(df_prompt_emotion_completion_emotion):
    """
    Check accuracy of the model
    """

    did_unique = list(df_prompt_emotion_completion_emotion['DID'].unique())
    emotion, emotion0, emotion1, emotion2, emotion3 = [], [], [], [], []
    emotion_completion, emotion_completion0, emotion_completion1, emotion_completion2, emotion_completion3 = [], [], [], [], []

    for did in did_unique:
        df_did = df_prompt_emotion_completion_emotion[df_prompt_emotion_completion_emotion['DID'] == did]
        df_did.reset_index(drop=True, inplace=True)
        emotion0.append(df_did['EMOTION'][0])
        emotion_completion0.append(df_did['EMOTION_COMPLETION'][0])
        emotion1.append(df_did['EMOTION'][1])
        emotion_completion1.append(df_did['EMOTION_COMPLETION'][1])
        emotion2.append(df_did['EMOTION'][2])
        emotion_completion2.append(df_did['EMOTION_COMPLETION'][2])
        emotion3.append(df_did['EMOTION'][3])
        emotion_completion3.append(df_did['EMOTION_COMPLETION'][3])
    emotion = df_prompt_emotion_completion_emotion['EMOTION'].to_list()
    emotion_completion = df_prompt_emotion_completion_emotion['EMOTION_COMPLETION'].to_list()

    results = [
        [emotion0, emotion_completion0],
        [emotion1, emotion_completion1],
        [emotion2, emotion_completion2],
        [emotion3, emotion_completion3],
        [emotion, emotion_completion]
    ]

    for idx, result in enumerate(results):
        get_metric(result[0], result[1], idx)
    nones = len(df_prompt_emotion_completion_emotion[df_prompt_emotion_completion_emotion['EMOTION_COMPLETION'] == 'none'])
    print(f'#nones: {nones}\n')
    
    count_values = df_prompt_emotion_completion_emotion['EMOTION_COMPLETION'].value_counts().sort_index(ascending=True)
    print(f'count_values: {count_values}')


##########################################
############### COMPLETION ###############
##########################################

def handle_message(msg):
    """
    Get completion
    """

    prompt = msg['text']
    
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=200,
        tok_k=0.0,
        top_p=1.0,
        temperature=0.7
        )

    model_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config
        )

    output = tokenizer.decode(
        model_output[0],
        skip_special_tokens=True
        )
    
    return output


def get_completion(df_prompt_emotion, prompt_prediction):
    """
    Get completions
    """
    
    model_outputs = []
    prompts = df_prompt_emotion['PROMPT'].to_list()
    first_completion = True
    for prompt in tqdm(prompts):
        message = {'text': [prompt]}
        completion = handle_message(message)

        if model_name_save in REPEATE_PROMPTS:
            pos = completion.rfind('A:')
            processed_completion = completion[pos+2:]
            
        elif model_name_save in REPEATE_PROMPTS_LLAMA3:
            prompt_split = prompt.splitlines()
            question = prompt_split[-3][:-55].replace(" ", "").lower()
            completion_split = completion.splitlines()

            for i, s in enumerate(completion_split):
                if question in s.replace(" ", "").lower():
                    completion_split = completion_split[i:]
                    break

            for i, s in enumerate(completion_split):
                if 'A: ' in s:
                    processed_completion = completion_split[i]
                    break

            processed_completion = processed_completion.replace(".assistant", "").replace(".user", "").replace("A:", "") + '.'
            
            # Only for LLAMA3 no instruct
            emotions_options = [
                    '[A] Anger',
                    '[B] Disgust',
                    '[C] Fear',
                    '[D] Happiness',
                    '[E] Sadness',
                    '[F] Surprise',
                    '[G] Neutral'
                    ]
            for e_o in emotions_options:
                pos_end = processed_completion.find(e_o)
                if pos_end != -1:
                    processed_completion = processed_completion[:pos_end+len(e_o)] + '.'
                    break

        elif model_name_save in REPEATE_PROMPTS_GEMMA:
            pos = completion.rfind('A:')
            completion_split = completion[pos+2:]

            emotions_options = [
                    '[A] Anger',
                    '[B] Disgust',
                    '[C] Fear',
                    '[D] Happiness',
                    '[E] Sadness',
                    '[F] Surprise',
                    '[G] Neutral'
                    ]
            for e_o in emotions_options:
                pos_end = completion_split.find(e_o)
                if pos_end != -1:
                    completion_split = completion_split[:pos_end+len(e_o)] + '.'
                    break

            processed_completion = completion_split
            
        elif model_name_save in REPEATE_PROMPTS_MISTRAL:
            prompt_split = prompt.splitlines()
            question = prompt_split[-3].replace(" ", "").lower()
            completion_split = completion.splitlines()

            for i, s in enumerate(completion_split):
                if question in s.replace(" ", "").lower():
                    processed_completion = completion_split[i+2]
                    break
            
            processed_completion = processed_completion.replace("A:", "").strip() + '.'
            processed_completion = processed_completion.replace("..", ".")

        elif model_name_save in REPEATE_PROMPTS_CHATGLM3 or model_name_save in REPEATE_PROMPTS_GLM4:
            prompt_split = prompt.splitlines()
            question = prompt_split[-2].replace(" ", "").lower()
            completion_split = completion.splitlines()

            for i, s in enumerate(completion_split):
                if question in s.replace(" ", "").lower():
                    try:
                        processed_completion = completion_split[i+2]
                        break
                    except:
                        completion_split = completion_split[i:]
                        for i, s in enumerate(completion_split):
                            if 'A: ' in s:
                                processed_completion = completion_split[i]
                                break
                        break

            processed_completion = processed_completion.replace("A:", "").strip()
            
        elif model_name_save in REPEATE_PROMPTS_BLOOMZ:
            pos = completion.rfind('A:')
            processed_completion = completion[pos+4:pos+5]
            
            emotions_options = [
                    '[A] Anger',
                    '[B] Disgust',
                    '[C] Fear',
                    '[D] Happiness',
                    '[E] Sadness',
                    '[F] Surprise',
                    '[G] Neutral'
                    ]
            for e_o in emotions_options:
                if e_o[1:2] == processed_completion:
                    processed_completion = e_o
                    break

        elif model_name_save in REPEATE_PROMPTS_PHI3:
            prompt_split = prompt.splitlines()[-3][:-7]
            pos = completion.find(prompt_split)
            completion_split = completion[pos+len(prompt_split):]
            pos = completion_split.find('Q:')
            processed_completion = completion_split[:pos] + '.'
            processed_completion = processed_completion.replace("A:", "")

        elif model_name_save in REPEATE_PROMPTS_AIRBOROS:
            pos = completion.rfind('A:')
            completion_split = completion[pos:]
            
            emotions_options = [
                    '[A] Anger',
                    '[B] Disgust',
                    '[C] Fear',
                    '[D] Happiness',
                    '[E] Sadness',
                    '[F] Surprise',
                    '[G] Neutral'
                    ]

            for e_o in emotions_options:
                pos_end = completion_split.find(e_o)
                if pos_end != -1:
                    processed_completion = completion_split[:pos_end+len(e_o)] + '.'
                    break
            
            processed_completion = processed_completion.replace("A:", "")

        elif model_name_save in REPEATE_PROMPTS_ZEPHYR_ALPHA:
            prompt_split = prompt.splitlines()
            question = prompt_split[-3][:-4].replace(" ", "").lower()
            completion_split = completion.splitlines()

            for i, s in enumerate(completion_split):
                if question in s.replace(" ", "").lower():
                    completion_split = completion_split[i+1:]
                    break

            completion_split = ' '.join(completion_split)
            pos = completion_split.find('<|user|>')
            processed_completion = completion_split[:pos]

            processed_completion = processed_completion.replace("<|assistant|>", "").replace("A:", "").strip() + '.'
            processed_completion = processed_completion.replace("..", ".")

        elif model_name_save in REPEATE_PROMPTS_ZEPHYR_BETA:
            prompt_split = prompt.splitlines()
            question = prompt_split[-3][:-4].replace(" ", "").lower()
            completion_split = completion.splitlines()

            for i, s in enumerate(completion_split):
                if question in s.replace(" ", "").lower():
                    completion_split = completion_split[i:]
                    break

            for i, s in enumerate(completion_split):
                if 'A: ' in s:
                    processed_completion = completion_split[i]
                    break

            processed_completion = processed_completion.replace("A:", "").strip() + '.'
            processed_completion = processed_completion.replace("..", ".")

        elif model_name_save in REPEATE_PROMPTS_ZEPHYR_GEMMA:
            prompt_split = prompt.splitlines()
            question = prompt_split[-3][:-4].replace(" ", "").lower()
            completion_split = completion.splitlines()

            for i, s in enumerate(completion_split):
                if question in s.replace(" ", "").replace("</s>", "").lower():
                    completion_split = completion_split[i+2:]
                    break

            for i, s in enumerate(completion_split):
                if 'the answer is' in s.lower():
                    processed_completion = completion_split[i]
                    break
            
            processed_completion = processed_completion.replace("A:", "").replace("</s>", "").strip()

        elif model_name_save in REPEATE_PROMPTS_INTERNLM:
            prompt_split = prompt.splitlines()
            question = prompt_split[-3][:-10].replace(" ", "").lower()
            completion_split = completion.splitlines()

            for i, s in enumerate(completion_split):
                if question in s.replace(" ", "").lower():
                    completion_split = completion_split[i:]
                    break

            for i, s in enumerate(completion_split):
                if 'A: ' in s:
                    processed_completion = completion_split[i]
                    break

            processed_completion = processed_completion.replace("A:", "").replace("<|im_end|>", "").strip() + '.'
            processed_completion = processed_completion.replace("...", ".").replace("..", ".")

        else:
            processed_completion = completion

        processed_completion = ''.join(processed_completion.splitlines()).strip()
        model_outputs.append(processed_completion)

        if first_completion:
            logging.info('\n\nFIRST PROMPT:\n' + prompt)
            logging.info('\n\nCOMPLETION:\n' + completion)
            logging.info('\n\nPROCESSED COMPLETION:\n' + processed_completion)
            first_completion = False
            if debug:
                first_completion = True

    df_prompt_emotion['COMPLETION'] = model_outputs

    logging.info('SAVING COMPLETIONS ' + str(prompt_prediction))
    df_prompt_emotion.to_csv('data/' + dataset + '/' + model_name_save + '/' + dataset + '_' + model_name_save + '_completions_' + str(prompt_prediction) + '.csv', index=False)

    return df_prompt_emotion


##########################################
################# PROMPT #################
##########################################

if model_name_save in INPUTTEMPLATE1:
    system = ''
    user = '\nUSER:'
    assistant = '\nASSISTANT:\n'
    eos = ''
elif model_name_save in INPUTTEMPLATE2:
    system = ''
    user = '\n### Instruction:'
    assistant = '\n### Response:\n'
    eos = ''
elif model_name_save in INPUTTEMPLATE3:
    system = '<|system|>'
    user = '<|user|>'
    assistant = '<|assistant|>'
    eos = '</s>'
elif model_name_save in INPUTTEMPLATELLAMA2:
    bos = '<s>'
    system_start = '<<SYS>>'
    system_end = '<</SYS>>'
    user = '[INST]'
    assistant = '[/INST]'
    eos = '</s>'
elif model_name_save in INPUTTEMPLATELLAMA3:
    bos = '<|begin_of_text|>'
    user = '<|start_header_id|>'
    assistant = '<|end_header_id|>'
    eos = '<|eot_id|>'
elif model_name_save in INPUTTEMPLATEGEMMA:
    user = '<start_of_turn>user'
    assistant = '<start_of_turn>model'
    eos = '<end_of_turn>'
elif model_name_save in INPUTTEMPLATEPHI3:
    user = '<|user|>'
    assistant = '<|assistant|>'
    eos = '<|end|>'
elif model_name_save in INPUTTEMPLATEMISTRAL:
    system = ''
    bos = '<s>'
    user = '[INST] '
    assistant = '[/INST]'
    eos = '</s>'
elif model_name_save in INPUTTEMPLATECHATGLM3:
    system = '<|system|>'
    user = '<|user|>'
    assistant = '<|assistant|>'
elif model_name_save in INPUTTEMPLATEGLM4:
    system = '<|system|>'
    user = '<|user|>'
    assistant = '<|assistant|>'
    eos = '<|endoftext|>'
elif model_name_save in INPUTTEMPLATEINTERLM:
    system = '<|im_start|>system'
    user = '<|im_start|>user'
    assistant = '<|im_start|>assistant'
    eos = '<|im_end|>'
else:
    system = ''
    user = ''
    assistant = ''
    eos = ''

role = f"""You are an emotional classifier."""

instruction = f"""Follow the logic of the following example. Be aware of the emotional context of the dialogue.
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.

Apply the logic learned in the example. The answer must end with one of the different possible options."""

options = f"""[A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral."""


##########################################
################ PROMPT 1 ################
##########################################

def make_prompt_1(df_dialogues):
    """
    Create the prompt 1 for the LLM
    """
      
    prompt_list = []
    df_dialogues_unique = df_dialogues['DID'].unique()
    for _, did in tqdm(enumerate(df_dialogues_unique)):
        dialogue = df_dialogues['SEG'][df_dialogues['DID'] == did].to_list()

        if model_name_save in INPUTTEMPLATELLAMA3:
            prompt = f"""{bos}{user}system{assistant}
        
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.{eos}{user}user{assistant}

Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}{user}assistant{assistant}

A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.{eos}{user}user{assistant}

Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}{user}assistant{assistant}

A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.{eos}{user}user{assistant}

Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}{user}assistant{assistant}

A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.{eos}{user}system{assistant}

Apply the logic learned in the example. The answer must end with one of the different possible options.{eos}{user}user{assistant}

Q: {dialogue[0]}{options}{eos}{user}assistant{assistant}

A:"""
        
        elif model_name_save in INPUTTEMPLATEGEMMA or model_name_save in INPUTTEMPLATEPHI3:
            prompt = f"""{user}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.{eos}
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.{eos}
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.{eos}
{user}
Apply the logic learned in the example. The answer must end with one of the different possible options.
Q: {dialogue[0]}{options}{eos}
{assistant}
A: """

        elif model_name_save in INPUTTEMPLATECHATGLM3:
            prompt = f"""{system}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
{user}
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.
{system}
Apply the logic learned in the example. The answer must end with one of the different possible options.
{user}
Q: {dialogue[0]}{options}
{assistant}"""
            
        elif model_name_save in INPUTTEMPLATEGLM4:
            prompt = f"""{system}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
{user}
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.
{system}
Apply the logic learned in the example. The answer must end with one of the different possible options.
{user}
Q: {dialogue[0]}{options}
{assistant}"""

        elif model_name_save in INPUTTEMPLATEMISTRAL:
            prompt = f"""{bos}{user}You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.
{eos}
{user}
Apply the logic learned in the example. The answer must end with one of the different possible options.
Q: {dialogue[0].strip()}{options}
{assistant}
A: """

        elif model_name_save in INPUTTEMPLATELLAMA2:
            prompt = f"""{bos}{user} {system_start}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
{system_end}

Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral. {eos}
{bos}{user} Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger. {eos}
{bos}{user} Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral. {eos}

{bos}{user} {system_start}
Apply the logic learned in the example. The answer must end with one of the different possible options.
{system_end}

Q: {dialogue[0]}{options} {assistant} A: """

        elif model_name_save in INPUTTEMPLATEINTERLM:
            prompt = f"""{system}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.{eos}
{user}
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.{eos}
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.{eos}
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.{eos}

{system}
Apply the logic learned in the example. The answer must end with one of the different possible options.
{user}
Q: {dialogue[0]}{options}{eos}
{assistant}
A: """

        elif model_name_save in INPUTTEMPLATE3:
            prompt = f"""{system}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.{eos}
{user}
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.{eos}
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.{eos}
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.{eos}

{system}
Apply the logic learned in the example. The answer must end with one of the different possible options.{eos}
{user}
Q: {dialogue[0]}{options}{eos}
{assistant}
A: """

        elif model_name_save in INPUTTEMPLATE1 or model_name_save in INPUTTEMPLATE2:
            prompt = f"""{bos}{user}{system_start}You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.{system_end}

Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral. {eos}
{bos} {user} Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger. {eos}
{bos} {user} Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral. {eos}

{bos}{user}{system_start}Apply the logic learned in the example. The answer must end with one of the different possible options.{system_end}

Q: {dialogue[0]}{options} {assistant} A: """

        else:
            prompt = f"""You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.

Apply the logic learned in the example. The answer must end with one of the different possible options.
Q: {dialogue[0]}{options}
A: """
        
        prompt_list.append(prompt)
  
    df_prompt_emotion = pd.DataFrame(prompt_list, columns = ['PROMPT'])
    return df_prompt_emotion


##########################################
################ PROMPT 2 ################
##########################################

def make_prompt_2(df_dialogues, df_completions_emotion_1):
    """
    Create the prompt 2 for the LLM
    """
      
    prompt_list = []
    df_dialogues_unique = df_dialogues['DID'].unique()
    for idx, did in tqdm(enumerate(df_dialogues_unique)):
        dialogue = df_dialogues['SEG'][df_dialogues['DID'] == did].to_list()
        completion_1 = df_completions_emotion_1['COMPLETION'][idx]
        
        if model_name_save in INPUTTEMPLATELLAMA3:
            prompt = f"""{bos}{user}system{assistant}
        
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.{eos}{user}user{assistant}

Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}{user}assistant{assistant}

A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.{eos}{user}user{assistant}

Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}{user}assistant{assistant}

A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.{eos}{user}user{assistant}

Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}{user}assistant{assistant}

A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.{eos}

{bos}{user}system{assistant}

Apply the logic learned in the example. The answer must end with one of the different possible options.{eos}{user}user{assistant}

Q: {dialogue[0]}{options}{eos}{user}assistant{assistant}

A: {completion_1}{eos}{user}user{assistant}

Q:{dialogue[1]}{options}{eos}{user}assistant{assistant}

A:"""

        elif model_name_save in INPUTTEMPLATEGEMMA or model_name_save in INPUTTEMPLATEPHI3:
            prompt = f"""{user}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.{eos}
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.{eos}
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.{eos}
{user}
Apply the logic learned in the example. The answer must end with one of the different possible options.
Q: {dialogue[0]}{options}{eos}
{assistant}
A: {completion_1}{eos}
{user}
Q: {dialogue[1].strip()}{options}{eos}
{assistant}
A:"""
        
        elif model_name_save in INPUTTEMPLATEMISTRAL:
            prompt = f"""{bos}{user}You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.
{eos}
{user}
Apply the logic learned in the example. The answer must end with one of the different possible options.
Q: {dialogue[0].strip()}{options}
A: {completion_1}
Q: {dialogue[1].strip()}{options}
{assistant}
A: """

        elif model_name_save in INPUTTEMPLATECHATGLM3:
            prompt = f"""{system}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
{user}
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.
{system}
Apply the logic learned in the example. The answer must end with one of the different possible options.
{user}
Q: {dialogue[0].strip()}{options}
{assistant}
A: {completion_1}
{user}
Q: {dialogue[1].strip()}{options}
{assistant}"""

        elif model_name_save in INPUTTEMPLATEGLM4:
            prompt = f"""{system}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
{user}
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.
{system}
Apply the logic learned in the example. The answer must end with one of the different possible options.
{user}
Q: {dialogue[0].strip()}{options}
{assistant}
A: {completion_1}
{user}
Q: {dialogue[1].strip()}{options}
{assistant}"""

        elif model_name_save in INPUTTEMPLATELLAMA2:
            prompt = f"""{bos}{user} {system_start}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
{system_end}

Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral. {eos}
{bos}{user} Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger. {eos}
{bos}{user} Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral. {eos}

{bos}{user} {system_start}
Apply the logic learned in the example. The answer must end with one of the different possible options.
{system_end}

Q: {dialogue[0]}{options} {assistant} A: {completion_1} {eos}
{bos}{user} Q: {dialogue[1]}{options} {assistant} A: """

        elif model_name_save in INPUTTEMPLATEINTERLM:
            prompt = f"""{system}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.{eos}
{user}
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.{eos}
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.{eos}
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.{eos}

{system}
Apply the logic learned in the example. The answer must end with one of the different possible options.
{user}
Q: {dialogue[0]}{options}{eos}
{assistant}
A: {completion_1}{eos}
{user}
Q: {dialogue[1].strip()} {options}{eos}
{assistant}
A: """

        elif model_name_save in INPUTTEMPLATE3:
            prompt = f"""{system}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.{eos}
{user}
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.{eos}
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.{eos}
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.{eos}

{system}
Apply the logic learned in the example. The answer must end with one of the different possible options.{eos}
{user}
Q: {dialogue[0]}{options}{eos}
{assistant}
A: {completion_1}{eos}
{user}
Q: {dialogue[1].strip()} {options}{eos}
{assistant}
A: """

        elif model_name_save in INPUTTEMPLATE1 or model_name_save in INPUTTEMPLATE2:
            prompt = f"""{bos}{user}{system_start}You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.{system_end}

Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral. {eos}
{bos} {user} Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger. {eos}
{bos} {user} Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral. {eos}

{bos}{user}{system_start}Apply the logic learned in the example. The answer must end with one of the different possible options.{system_end}

Q: {dialogue[0]}{options} {assistant} A: {completion_1} {eos}
{bos} {user} Q: {dialogue[1]}{options} {assistant} A: """
            
        else:
            prompt = f"""You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.

Apply the logic learned in the example. The answer must end with one of the different possible options.
Q: {dialogue[0]}{options}
A: {completion_1}
Q: {dialogue[1].strip()}{options}
A: """
        
        prompt_list.append(prompt)
  
    df_prompt_emotion = pd.DataFrame(prompt_list, columns = ['PROMPT'])
    return df_prompt_emotion


##########################################
################ PROMPT 3 ################
##########################################

def make_prompt_3(df_dialogues, df_completions_emotion_1, df_completions_emotion_2):
    """
    Create the prompt for the LLM
    """
      
    prompt_list = []
    df_dialogues_unique = df_dialogues['DID'].unique()
    for idx, did in tqdm(enumerate(df_dialogues_unique)):
        dialogue = df_dialogues['SEG'][df_dialogues['DID'] == did].to_list()
        completion_1 = df_completions_emotion_1['COMPLETION'][idx]
        completion_2 = df_completions_emotion_2['COMPLETION'][idx]
        
        if model_name_save in INPUTTEMPLATELLAMA3:
            prompt = f"""{bos}{user}system{assistant}
        
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.{eos}{user}user{assistant}

Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}{user}assistant{assistant}

A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.{eos}{user}user{assistant}

Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}{user}assistant{assistant}

A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.{eos}{user}user{assistant}

Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}{user}assistant{assistant}

A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.{eos}

{bos}{user}system{assistant}

Apply the logic learned in the example. The answer must end with one of the different possible options.{eos}{user}user{assistant}

Q: {dialogue[0]}{options}{eos}{user}assistant{assistant}

A: {completion_1}{eos}{user}user{assistant}

Q:{dialogue[1]}{options}{eos}{user}assistant{assistant}

A: {completion_2}{eos}{user}user{assistant}

Q:{dialogue[2]}{options}{eos}{user}assistant{assistant}

A:"""

        elif model_name_save in INPUTTEMPLATEGEMMA or model_name_save in INPUTTEMPLATEPHI3:
            prompt = f"""{user}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.{eos}
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.{eos}
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.{eos}
{user}
Apply the logic learned in the example. The answer must end with one of the different possible options.
Q: {dialogue[0]}{options}{eos}
{assistant}
A: {completion_1}{eos}
{user}
Q: {dialogue[1].strip()}{options}{eos}
{assistant}
A: {completion_2}{eos}
{user}
Q: {dialogue[2].strip()}{options}{eos}
{assistant}
A:"""
            
        elif model_name_save in INPUTTEMPLATEMISTRAL:
            prompt = f"""{bos}{user}You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.
{eos}
{user}
Apply the logic learned in the example. The answer must end with one of the different possible options.
Q: {dialogue[0].strip()}{options}
A: {completion_1}
Q: {dialogue[1].strip()}{options}
A: {completion_2}
Q: {dialogue[2].strip()}{options}
{assistant}
A: """

        elif model_name_save in INPUTTEMPLATECHATGLM3:
            prompt = f"""{system}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
{user}
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.
{system}
Apply the logic learned in the example. The answer must end with one of the different possible options.
{user}
Q: {dialogue[0].strip()}{options}
{assistant}
A: {completion_1}
{user}
Q: {dialogue[1].strip()}{options}
{assistant}
A: {completion_2}
{user}
Q: {dialogue[2].strip()}{options}
{assistant}"""
            
        elif model_name_save in INPUTTEMPLATEGLM4:
            prompt = f"""{system}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
{user}
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.
{system}
Apply the logic learned in the example. The answer must end with one of the different possible options.
{user}
Q: {dialogue[0].strip()}{options}
{assistant}
A: {completion_1}
{user}
Q: {dialogue[1].strip()}{options}
{assistant}
A: {completion_2}
{user}
Q: {dialogue[2].strip()}{options}
{assistant}"""

        elif model_name_save in INPUTTEMPLATELLAMA2:
            prompt = f"""{bos}{user} {system_start}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
{system_end}

Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral. {eos}
{bos}{user} Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger. {eos}
{bos}{user} Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral. {eos}

{bos}{user} {system_start}
Apply the logic learned in the example. The answer must end with one of the different possible options.
{system_end}

Q: {dialogue[0]}{options} {assistant} A: {completion_1} {eos}
{bos}{user} Q: {dialogue[1]}{options} {assistant} A: {completion_2} {eos}
{bos}{user} Q: {dialogue[2]}{options} {assistant} A: """

        elif model_name_save in INPUTTEMPLATEINTERLM:
            prompt = f"""{system}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.{eos}
{user}
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.{eos}
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.{eos}
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.{eos}

{system}
Apply the logic learned in the example. The answer must end with one of the different possible options.
{user}
Q: {dialogue[0]}{options}{eos}
{assistant}
A: {completion_1}{eos}
{user}
Q: {dialogue[1].strip()} {options}{eos}
{assistant}
A: {completion_2}{eos}
{user}
Q: {dialogue[2].strip()} {options}{eos}
{assistant}
A: """

        elif model_name_save in INPUTTEMPLATE3:
            prompt = f"""{system}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.{eos}
{user}
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.{eos}
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.{eos}
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.{eos}

{system}
Apply the logic learned in the example. The answer must end with one of the different possible options.{eos}
{user}
Q: {dialogue[0]}{options}{eos}
{assistant}
A: {completion_1}{eos}
{user}
Q: {dialogue[1].strip()} {options}{eos}
{assistant}
A: {completion_2}{eos}
{user}
Q: {dialogue[2].strip()} {options}{eos}
{assistant}
A: """

        elif model_name_save in INPUTTEMPLATE1 or model_name_save in INPUTTEMPLATE2:
            prompt = f"""{bos}{user}{system_start}You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.{system_end}

Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral. {eos}
{bos} {user} Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger. {eos}
{bos} {user} Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral. {eos}

{bos}{user}{system_start}Apply the logic learned in the example. The answer must end with one of the different possible options.{system_end}

Q: {dialogue[0]}{options} {assistant} A: {completion_1} {eos}
{bos} {user} Q: {dialogue[1]}{options} {assistant} A: {completion_2} {eos}
{bos} {user} Q: {dialogue[2]}{options} {assistant} A: """
            
        else:
            prompt = f"""You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.

Apply the logic learned in the example. The answer must end with one of the different possible options.
Q: {dialogue[0]}{options}
A: {completion_1}
Q: {dialogue[1].strip()}{options}
A: {completion_2}
Q: {dialogue[2].strip()}{options}
A: """
        
        prompt_list.append(prompt)
    df_prompt_emotion = pd.DataFrame(prompt_list, columns = ['PROMPT'])
    return df_prompt_emotion


##########################################
################ PROMPT 4 ################
##########################################

def make_prompt_4(df_dialogues, df_completions_emotion_1, df_completions_emotion_2, df_completions_emotion_3):
    """
    Create the prompt for the LLM
    """
      
    prompt_list = []
    df_dialogues_unique = df_dialogues['DID'].unique()
    for idx, did in tqdm(enumerate(df_dialogues_unique)):
        dialogue = df_dialogues['SEG'][df_dialogues['DID'] == did].to_list()
        completion_1 = df_completions_emotion_1['COMPLETION'][idx]
        completion_2 = df_completions_emotion_2['COMPLETION'][idx]
        completion_3 = df_completions_emotion_3['COMPLETION'][idx]
        
        if model_name_save in INPUTTEMPLATELLAMA3:
            prompt = f"""{bos}{user}system{assistant}
        
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.{eos}{user}user{assistant}

Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}{user}assistant{assistant}

A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.{eos}{user}user{assistant}

Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}{user}assistant{assistant}

A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.{eos}{user}user{assistant}

Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}{user}assistant{assistant}

A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.{eos}

{bos}{user}system{assistant}

Apply the logic learned in the example. The answer must end with one of the different possible options.{eos}{user}user{assistant}

Q: {dialogue[0]}{options}{eos}{user}assistant{assistant}

A: {completion_1}{eos}{user}user{assistant}

Q:{dialogue[1]}{options}{eos}{user}assistant{assistant}

A: {completion_2}{eos}{user}user{assistant}

Q:{dialogue[2]}{options}{eos}{user}assistant{assistant}

A: {completion_3}{eos}{user}user{assistant}

Q:{dialogue[3]}{options}{eos}{user}assistant{assistant}

A:"""
        
        elif model_name_save in INPUTTEMPLATEGEMMA or model_name_save in INPUTTEMPLATEPHI3:
            prompt = f"""{user}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.{eos}
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.{eos}
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.{eos}
{user}
Apply the logic learned in the example. The answer must end with one of the different possible options.
Q: {dialogue[0]}{options}{eos}
{assistant}
A: {completion_1}{eos}
{user}
Q: {dialogue[1].strip()}{options}{eos}
{assistant}
A: {completion_2}{eos}
{user}
Q: {dialogue[2].strip()}{options}{eos}
{assistant}
A: {completion_3}{eos}
{user}
Q: {dialogue[3].strip()}{options}{eos}
{assistant}
A:"""
            
        elif model_name_save in INPUTTEMPLATEMISTRAL:
            prompt = f"""{bos}{user}You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.
{eos}
{user}
Apply the logic learned in the example. The answer must end with one of the different possible options.
Q: {dialogue[0].strip()}{options}
A: {completion_1}
Q: {dialogue[1].strip()}{options}
A: {completion_2}
Q: {dialogue[2].strip()}{options}
A: {completion_3}
Q: {dialogue[3].strip()}{options}
{assistant}
A: """

        elif model_name_save in INPUTTEMPLATECHATGLM3:
            prompt = f"""{system}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
{user}
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.
{system}
Apply the logic learned in the example. The answer must end with one of the different possible options.
{user}
Q: {dialogue[0].strip()}{options}
{assistant}
A: {completion_1}
{user}
Q: {dialogue[1].strip()}{options}
{assistant}
A: {completion_2}
{user}
Q: {dialogue[2].strip()}{options}
{assistant}
A: {completion_3}
{user}
Q: {dialogue[3].strip()}{options}
{assistant}"""
            
        elif model_name_save in INPUTTEMPLATEGLM4:
            prompt = f"""{system}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
{user}
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.
{system}
Apply the logic learned in the example. The answer must end with one of the different possible options.
{user}
Q: {dialogue[0].strip()}{options}
{assistant}
A: {completion_1}
{user}
Q: {dialogue[1].strip()}{options}
{assistant}
A: {completion_2}
{user}
Q: {dialogue[2].strip()}{options}
{assistant}
A: {completion_3}
{user}
Q: {dialogue[3].strip()}{options}
{assistant}"""
            
        elif model_name_save in INPUTTEMPLATELLAMA2:
            prompt = f"""{bos}{user} {system_start}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
{system_end}

Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral. {eos}
{bos}{user} Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger. {eos}
{bos}{user} Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral. {eos}

{bos}{user} {system_start}
Apply the logic learned in the example. The answer must end with one of the different possible options.
{system_end}

Q: {dialogue[0]}{options} {assistant} A: {completion_1} {eos}
{bos}{user} Q: {dialogue[1]}{options} {assistant} A: {completion_2} {eos}
{bos}{user} Q: {dialogue[2]}{options} {assistant} A: {completion_3} {eos}
{bos}{user} Q: {dialogue[3]}{options} {assistant} A: """

        elif model_name_save in INPUTTEMPLATEINTERLM:
            prompt = f"""{system}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.{eos}
{user}
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.{eos}
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.{eos}
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.{eos}

{system}
Apply the logic learned in the example. The answer must end with one of the different possible options.
{user}
Q: {dialogue[0]}{options}{eos}
{assistant}
A: {completion_1}{eos}
{user}
Q: {dialogue[1].strip()} {options}{eos}
{assistant}
A: {completion_2}{eos}
{user}
Q: {dialogue[2].strip()} {options}{eos}
{assistant}
A: {completion_3}{eos}
{user}
Q: {dialogue[3].strip()} {options}{eos}
{assistant}
A: """

        elif model_name_save in INPUTTEMPLATE3:
            prompt = f"""{system}
You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.{eos}
{user}
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.{eos}
{user}
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.{eos}
{user}
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.{eos}
{assistant}
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.{eos}

{system}
Apply the logic learned in the example. The answer must end with one of the different possible options.{eos}
{user}
Q: {dialogue[0]}{options}{eos}
{assistant}
A: {completion_1}{eos}
{user}
Q: {dialogue[1].strip()} {options}{eos}
{assistant}
A: {completion_2}{eos}
{user}
Q: {dialogue[2].strip()} {options}{eos}
{assistant}
A: {completion_3}{eos}
{user}
Q: {dialogue[3].strip()} {options}{eos}
{assistant}
A: """

        elif model_name_save in INPUTTEMPLATE1 or model_name_save in INPUTTEMPLATE2:
            prompt = f"""{bos}{user}{system_start}You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.{system_end}

Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral. {eos}
{bos} {user} Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger. {eos}
{bos} {user} Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral. {assistant} A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral. {eos}

{bos}{user}{system_start}Apply the logic learned in the example. The answer must end with one of the different possible options.{system_end}

Q: {dialogue[0]}{options} {assistant} A: {completion_1} {eos}
{bos} {user} Q: {dialogue[1]}{options} {assistant} A: {completion_2} {eos}
{bos} {user} Q: {dialogue[2]}{options} {assistant} A: {completion_3} {eos}
{bos} {user} Q: {dialogue[3]}{options} {assistant} A: """

        else:
            prompt = f"""You are an emotional classifier. Follow the logic of the following example. Be aware of the emotional context of the dialogue.
Q: Are things still going badly with your houseguest? [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.
Q: Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.
Q: Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law. [A] Anger, [B] Disgust, [C] Fear, [D] Happiness, [E] Sadness, [F] Surprise or [G] Neutral.
A: Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.

Apply the logic learned in the example. The answer must end with one of the different possible options.
Q: {dialogue[0]}{options}
A: {completion_1}
Q: {dialogue[1].strip()}{options}
A: {completion_2}
Q: {dialogue[2].strip()}{options}
A: {completion_3}
Q: {dialogue[3].strip()}{options}
A: """
        
        prompt_list.append(prompt)
      
    df_prompt_emotion = pd.DataFrame(prompt_list, columns = ['PROMPT'])
    return df_prompt_emotion


##########################################
################## MAIN ##################
##########################################

## PRE-PROCESSING ##
logging.info('LOADING DIALOGUES')
df_dialogues = create_dialogues()
# UNCOMMENT ONLY FOR TEST FEW DIALOGUES
if debug:
    turns, dialogs = 4, 5
    df_dialogues = df_dialogues[:turns*dialogs]

## FIRST STEP ##
logging.info('CREATING PROMPTS 1')
df_prompt_emotion = make_prompt_1(df_dialogues)
logging.info('GENERATING COMPLETIONS 1')
df_prompt_emotion_completion_1 = get_completion(df_prompt_emotion, 1)
## SECOND STEP ##
logging.info('CREATING PROMPTS 2')
df_prompt_emotion = make_prompt_2(df_dialogues, df_prompt_emotion_completion_1)
logging.info('GENERATING COMPLETIONS 2')
df_prompt_emotion_completion_2 = get_completion(df_prompt_emotion, 2)
## THIRD STEP ##
logging.info('CREATING PROMPTS 3')
df_prompt_emotion = make_prompt_3(df_dialogues, df_prompt_emotion_completion_1, df_prompt_emotion_completion_2)
logging.info('GENERATING COMPLETIONS 3')
df_prompt_emotion_completion_3 = get_completion(df_prompt_emotion, 3)
## FOURTH STEP ##
logging.info('CREATING PROMPTS 4')
df_prompt_emotion = make_prompt_4(df_dialogues, df_prompt_emotion_completion_1, df_prompt_emotion_completion_2, df_prompt_emotion_completion_3)
logging.info('GENERATING COMPLETIONS 4')
df_prompt_emotion_completion_4 = get_completion(df_prompt_emotion, 4)

## POST-PROCESSING ##
logging.info('PROCESSING EMOTIONS FROM COMPLETIONS')
df_prompt_emotion_completion_emotion = get_emotion_completion(
    df_dialogues,
    df_prompt_emotion_completion_1,
    df_prompt_emotion_completion_2,
    df_prompt_emotion_completion_3,
    df_prompt_emotion_completion_4
    )

## EVALUATE ##
logging.info('CHECKING ACCURACY')
check_accuracy(df_prompt_emotion_completion_emotion)

# logging.info('CHECKING COMPLETIONS EMOTIONS')
# check_valid_emotions(df_prompt_emotion_completion_emotion)
