#!/bin/bash
# GenEval Image Generation - With Reasoning
# Usage: bash scripts/generate_geneval_with_reasoning.sh

DIFFUSION_MODEL="qwen_image"
MODEL_PATH="Qwen/Qwen-Image-2512"
METADATA_FILE="data/enhanced/geneval_enhanced.jsonl"
OUTPUT_DIR="data/generated_images/geneval_results"

python -m src.benchmark.geneval_generate \
    --diffusion_model ${DIFFUSION_MODEL} \
    --model_path ${MODEL_PATH} \
    --metadata_file ${METADATA_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --method with_reasoning \
    --num_inference_steps 50 \
    --guidance_scale 4.0 \
    --seed 42 \
    --n_samples 4 \
    --batch_size 1 \
    --height 1024 \
    --width 1024