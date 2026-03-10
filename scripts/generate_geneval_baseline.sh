#!/bin/bash
# GenEval Image Generation - Baseline (Multi-GPU)
# Usage: bash scripts/generate_geneval_baseline.sh

NUM_GPUS=8
DIFFUSION_MODEL="qwen_image"
MODEL_PATH="/ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen-Image"
METADATA_FILE="data/enhanced/geneval_enhanced.jsonl"
OUTPUT_DIR="data/generated_images/geneval_results"

accelerate launch --num_processes ${NUM_GPUS} \
    -m src.benchmark.geneval_generate \
    --diffusion_model ${DIFFUSION_MODEL} \
    --model_path ${MODEL_PATH} \
    --metadata_file ${METADATA_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --method baseline \
    --num_inference_steps 50 \
    --guidance_scale 4.0 \
    --seed 42 \
    --n_samples 4 \
    --batch_size 1 \
    --height 1328 \
    --width 1328