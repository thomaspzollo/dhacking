#!/bin/bash

# List of models to use for generating summaries
models=(
    "microsoft/Phi-3-mini-4k-instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Llama-2-7b-chat-hf"
    "mistralai/Mistral-7B-Instruct-v0.1"
    "mistralai/Mistral-7B-Instruct-v0.2"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "meta-llama/Meta-Llama-3-70B-Instruct"
    "google/gemma-2-9b-it"
    "google/gemma-2-2b-it"
    "Qwen/Qwen2-7B-Instruct"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

# Python script to run
script="generate_summaries.py"

# Loop through each model and call the Python script with the model as an argument
for model in "${models[@]}"; do
    echo "Running script with model: $model"
    python $script --gen_model="$model" --job="Police Officer"
    python $script --gen_model="$model" --job="Social Worker"
done