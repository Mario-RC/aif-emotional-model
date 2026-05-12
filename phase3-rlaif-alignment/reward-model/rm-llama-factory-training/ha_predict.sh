#!/bin/bash

# GEMMA2 - google/gemma-2-9b-it
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/gemma2/gemma-2-9b-it_lora_reward_predict_ha_1ep.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/gemma2/gemma-2-9b-it_lora_reward_predict_ha_2ep.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/gemma2/gemma-2-9b-it_lora_reward_predict_ha_3ep.yaml

# GLM4 - THUDM/glm-4-9b-chat-1m
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/glm4/glm-4-9b-chat-1m_lora_reward_predict_ha_1ep.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/glm4/glm-4-9b-chat-1m_lora_reward_predict_ha_2ep.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/glm4/glm-4-9b-chat-1m_lora_reward_predict_ha_3ep.yaml

# LLAMA3 - meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama3/Meta-Llama-3-8B-Instruct_lora_reward_predict_ha_1ep.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama3/Meta-Llama-3-8B-Instruct_lora_reward_predict_ha_2ep.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama3/Meta-Llama-3-8B-Instruct_lora_reward_predict_ha_3ep.yaml

# MISTRAL - mistralai/Mistral-7B-Instruct-v0.3
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/mistral/Mistral-7B-Instruct-v0.3_lora_reward_predict_ha_1ep.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/mistral/Mistral-7B-Instruct-v0.3_lora_reward_predict_ha_2ep.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/mistral/Mistral-7B-Instruct-v0.3_lora_reward_predict_ha_3ep.yaml

# PHI3 - microsoft/Phi-3-small-8k-instruct
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/phi3/Phi-3-small-8k-instruct_lora_reward_predict_ha_1ep.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/phi3/Phi-3-small-8k-instruct_lora_reward_predict_ha_2ep.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/phi3/Phi-3-small-8k-instruct_lora_reward_predict_ha_3ep.yaml
