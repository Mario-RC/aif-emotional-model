# Emotion-driven Conversational Agent

1. Introduce a novel dataset creation methodology that leverages advanced language models to generate diverse emotional interactions.
2. Extend AI feedback paradigms specifically for emotional dialogue generation by incorporating multiple expert LLM-based evaluators.
3. Propose a human-chatbot emotional, empathetic and engagement alignment framework that captures different intensity levels of expression.
4. Present an integrated approach that combines SFT with RLAIF-PPO or DPO to ensure both emotional consistency and sustained conversational engagement.

# Usage
## Evaluate models
```bash
python3 foundation_model_selection.py
```

## Generate AI Rating
A configuration file with the API_KEY is required for each model.
```bash
python3 generate_rating_data.py
```

# List of Models Evaluated
* declare-lab/flan-alpaca-xl
* declare-lab/flan-sharegpt-xl
* declare-lab/flan-gpt4all-xl
* declare-lab/flan-alpaca-gpt4-xl
* lmsys/vicuna-7b-v1.5
* WizardLM/WizardLM-7B-V1.0
* facebook/opt-iml-max-1.3b
* databricks/dolly-v2-7b
* tiiuae/falcon-7b-instruct
* JosephusCheung/Guanaco
* openaccess-ai-collective/minotaur-7b
* NousResearch/Nous-Hermes-llama-2-7b
* allenai/tulu-7b
* yulan-team/YuLan-Chat-2-13b-fp16
* fnlp/moss-moon-003-sft
* openbmb/UltraLM-13b
* openbmb/UltraLM-13b-v2.0
* Norquinal/llama-2-7b-claude-chat
* Norquinal/llama-2-7b-claude-instruct
* lmsys/fastchat-t5-3b-v1.0
* meta-llama/Llama-2-7b-hf
* meta-llama/Llama-2-7b-chat-hf
* meta-llama/Meta-Llama-3-8B
* meta-llama/Meta-Llama-3-8B-Instruct
* bigscience/bloomz-3b
* bigscience/bloomz-7b1
* jondurbin/airoboros-l2-7b-2.2.1
* jondurbin/airoboros-l2-7b-3.0
* jondurbin/airoboros-l2-7b-gpt4-2.0
* jondurbin/airoboros-l2-7b-gpt4-m2.0
* HuggingFaceH4/zephyr-7b-alpha
* HuggingFaceH4/zephyr-7b-beta
* HuggingFaceH4/zephyr-7b-gemma-v0.1
* mistralai/Mistral-7B-Instruct-v0.1
* mistralai/Mistral-7B-Instruct-v0.3
* THUDM/chatglm2-6b
* THUDM/chatglm3-6b
* THUDM/chatglm3-6b-32k
* THUDM/chatglm3-6b-128k
* THUDM/glm-4-9b
* THUDM/glm-4-9b-chat
* THUDM/glm-4-9b-chat-1m
* google/gemma-1.1-7b-it
* google/gemma-2-9b-it
* microsoft/Phi-3-mini-4k-instruct
* microsoft/Phi-3-mini-128k-instruct
* microsoft/Phi-3-small-8k-instruct
* microsoft/Phi-3-small-128k-instruct
* internlm/internlm2-7b
* internlm/internlm2-chat-7b
* internlm/internlm2_5-7b-chat
* internlm/internlm2_5-7b-chat-1m
