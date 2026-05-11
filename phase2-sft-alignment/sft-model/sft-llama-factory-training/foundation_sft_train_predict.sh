#!/bin/bash

# CHATGLM3 - THUDM/chatglm3-6b-32k
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/chatglm3/chatglm3-6b-32k_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/chatglm3/chatglm3-6b-32k_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/chatglm3/chatglm3-6b-32k_lora_sft.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/chatglm3/chatglm3-6b-32k_lora_predict.yaml

# GEMMA - google/gemma-1.1-7b-it
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/gemma1/gemma-1.1-7b-it_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/gemma1/gemma-1.1-7b-it_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/gemma1/gemma-1.1-7b-it_lora_sft.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/gemma1/gemma-1.1-7b-it_lora_predict.yaml

# GEMMA - google/gemma-2-9b-it
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/gemma2/gemma-2-9b-it_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/gemma2/gemma-2-9b-it_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/gemma2/gemma-2-9b-it_lora_sft.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/gemma2/gemma-2-9b-it_lora_predict.yaml

# GLM4 - THUDM/glm-4-9b-chat
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/glm4/glm-4-9b-chat-1m_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/glm4/glm-4-9b-chat-1m_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/glm4/glm-4-9b-chat-1m_lora_sft.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/glm4/glm-4-9b-chat-1m_lora_predict.yaml

# INTERNLM2 - internlm/internlm2-chat-7b
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/internlm2/internlm2-chat-7b_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/internlm2/internlm2-chat-7b_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/internlm2/internlm2-chat-7b_lora_sft.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/internlm2/internlm2-chat-7b_lora_predict.yaml

# INTERNLM2 - internlm/internlm2_5-7b-chat
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/internlm25/internlm2_5-7b-chat_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/internlm25/internlm2_5-7b-chat_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/internlm25/internlm2_5-7b-chat_lora_sft.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/internlm25/internlm2_5-7b-chat_lora_predict.yaml

# LLAMA2 - meta-llama/Llama-2-7b-chat-hf
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama2/Llama-2-7b-chat-hf_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama2/Llama-2-7b-chat-hf_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama2/Llama-2-7b-chat-hf_lora_sft.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama2/Llama-2-7b-chat-hf_lora_predict.yaml

# LLAMA3 - meta-llama/Meta-Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama3/Meta-Llama-3-8B-Instruct_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama3/Meta-Llama-3-8B-Instruct_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama3/Meta-Llama-3-8B-Instruct_lora_sft.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/llama3/Meta-Llama-3-8B-Instruct_lora_predict.yaml

# MISTRAL - mistralai/Mistral-7B-Instruct-v0.3
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/mistral/Mistral-7B-Instruct-v0.3_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/mistral/Mistral-7B-Instruct-v0.3_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/mistral/Mistral-7B-Instruct-v0.3_lora_sft.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/mistral/Mistral-7B-Instruct-v0.3_lora_predict.yaml

# PHI3 - microsoft/Phi-3-mini-4k-instruct
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/phi3mini/Phi-3-mini-4k-instruct_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/phi3mini/Phi-3-mini-4k-instruct_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/phi3mini/Phi-3-mini-4k-instruct_lora_sft.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/phi3mini/Phi-3-mini-4k-instruct_lora_predict.yaml

# PHI3 - microsoft/Phi-3-small-8k-instruct
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/phi3/Phi-3-small-8k-instruct_foundation.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/phi3/Phi-3-small-8k-instruct_foundation_predict.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/phi3/Phi-3-small-8k-instruct_lora_sft.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/phi3/Phi-3-small-8k-instruct_lora_predict.yaml
