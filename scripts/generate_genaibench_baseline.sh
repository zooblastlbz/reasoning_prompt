#!/bin/bash
# GenAI-Bench Image Generation - Baseline (Multi-GPU)
# Usage: bash scripts/generate_genaibench_baseline.sh

NUM_GPUS=8
DIFFUSION_MODEL="qwen_image"
MODEL_PATH="/ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen-Image"
METADATA_FILE="data/enhanced/genaibench_enhanced.json"
OUTPUT_DIR="data/generated_images/genaibench_results"

accelerate launch --num_processes ${NUM_GPUS} \
    -m src.benchmark.genaibench_generate \
    --diffusion_model ${DIFFUSION_MODEL} \
    --model_path ${MODEL_PATH} \
    --metadata_file ${METADATA_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --method baseline \
    --num_inference_steps 50 \
    --guidance_scale 4.0 \
    --seed 42 \
    --n_samples 1 \
    --batch_size 1 \
    --height 1328 \
    --width 1328