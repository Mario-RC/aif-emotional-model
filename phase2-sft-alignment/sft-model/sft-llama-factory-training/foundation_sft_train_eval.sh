#!/bin/bash

# CHATGLM3 - THUDM/chatglm3-6b-32k
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/chatglm3-6b-32k_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/chatglm3-6b-32k_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/chatglm3-6b-32k_lora_sft.yaml

# GEMMA - google/gemma-1.1-7b-it
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/gemma-1.1-7b-it_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/gemma-1.1-7b-it_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/gemma-1.1-7b-it_lora_sft.yaml

# GEMMA - google/gemma-2-9b-it
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/gemma-2-9b-it_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/gemma-2-9b-it_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/gemma-2-9b-it_lora_sft.yaml

# GLM4 - THUDM/glm-4-9b-chat
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/glm-4-9b-chat-1m_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/glm-4-9b-chat-1m_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/glm-4-9b-chat-1m_lora_sft.yaml

# INTERNLM2 - internlm/internlm2-chat-7b
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/internlm2-chat-7b_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/internlm2-chat-7b_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/internlm2-chat-7b_lora_sft.yaml

# INTERNLM2 - internlm/internlm2_5-7b-chat
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/internlm2_5-7b-chat_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/internlm2_5-7b-chat_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/internlm2_5-7b-chat_lora_sft.yaml

# LLAMA2 - meta-llama/Llama-2-7b-chat-hf
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/Llama-2-7b-chat-hf_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/Llama-2-7b-chat-hf_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/Llama-2-7b-chat-hf_lora_sft.yaml

# LLAMA3 - meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/Meta-Llama-3-8B-Instruct_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/Meta-Llama-3-8B-Instruct_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/Meta-Llama-3-8B-Instruct_lora_sft.yaml

# MISTRAL - mistralai/Mistral-7B-Instruct-v0.3
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/Mistral-7B-Instruct-v0.3_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/Mistral-7B-Instruct-v0.3_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/Mistral-7B-Instruct-v0.3_lora_sft.yaml

# PHI3 - microsoft/Phi-3-mini-4k-instruct
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/Phi-3-mini-4k-instruct_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/Phi-3-mini-4k-instruct_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/Phi-3-mini-4k-instruct_lora_sft.yaml

# PHI3 - microsoft/Phi-3-small-8k-instruct
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/Phi-3-small-8k-instruct_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/Phi-3-small-8k-instruct_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli chat examples/inference/Phi-3-small-8k-instruct_lora_sft.yaml
