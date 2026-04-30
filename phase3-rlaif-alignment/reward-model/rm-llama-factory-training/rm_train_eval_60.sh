#!/bin/bash

# GEMMA2 - google/gemma-2-9b-it
# CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/train_lora/gemma2/gemma-2-9b-it_lora_reward_1.yaml

# CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/train_lora/gemma2/gemma-2-9b-it_lora_reward_2.yaml
# CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/train_lora/gemma2/gemma-2-9b-it_lora_reward_predict_2.yaml

# CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/train_lora/gemma2/gemma-2-9b-it_lora_reward_3.yaml
# CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/train_lora/gemma2/gemma-2-9b-it_lora_reward_predict_3.yaml

# CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/train_lora/gemma2/gemma-2-9b-it_lora_reward_4.yaml
# CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/train_lora/gemma2/gemma-2-9b-it_lora_reward_predict_4.yaml

# CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/train_lora/gemma2/gemma-2-9b-it_lora_reward_predict.yaml
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/gemma2/gemma-2-9b-it_lora_reward.yaml

# GLM4 - THUDM/glm-4-9b-chat-1m
# CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/train_lora/glm4/glm-4-9b-chat-1m_lora_reward_1.yaml

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/glm4/glm-4-9b-chat-1m_lora_reward_2.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/glm4/glm-4-9b-chat-1m_lora_reward_predict_2.yaml

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/glm4/glm-4-9b-chat-1m_lora_reward_3.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/glm4/glm-4-9b-chat-1m_lora_reward_predict_3.yaml

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/glm4/glm-4-9b-chat-1m_lora_reward_4.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/glm4/glm-4-9b-chat-1m_lora_reward_predict_4.yaml

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/gemma2/gemma-2-9b-it_lora_reward_4.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/gemma2/gemma-2-9b-it_lora_reward_predict_4.yaml

# CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/train_lora/glm4/glm-4-9b-chat-1m_lora_predict.yaml
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/glm4/glm-4-9b-chat-1m_lora_reward.yaml

# LLAMA3 - meta-llama/Meta-Llama-3-8B-Instruct
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama3/Meta-Llama-3-8B-Instruct_lora_reward_1.yaml
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama3/Meta-Llama-3-8B-Instruct_lora_reward_2.yaml
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama3/Meta-Llama-3-8B-Instruct_lora_reward_3.yaml
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama3/Meta-Llama-3-8B-Instruct_lora_reward_4.yaml
# CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/train_lora/llama3/Meta-Llama-3-8B-Instruct_lora_predict.yaml
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/llama3/Meta-Llama-3-8B-Instruct_lora_reward.yaml

# MISTRAL - mistralai/Mistral-7B-Instruct-v0.3
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/mistral/Mistral-7B-Instruct-v0.3_lora_reward_1.yaml
# CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/train_lora/mistral/Mistral-7B-Instruct-v0.3_lora_reward_2.yaml
# CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/train_lora/mistral/Mistral-7B-Instruct-v0.3_lora_reward_3.yaml
# CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/train_lora/mistral/Mistral-7B-Instruct-v0.3