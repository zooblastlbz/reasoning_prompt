#!/bin/bash
# GenAI-Bench Prompt Enhancement
# Usage: bash scripts/enhance_genaibench.sh

ENHANCER="hunyuan"
BACKEND="vllm"
ENHANCER_MODEL_PATH="/ytech_m2v5_hdd/workspace/kling_mm/Models/Hunyuan-Enhancer"
INPUT_FILE="data/prompts/genaibench_metadata.json"
OUTPUT_FILE="data/enhanced/genaibench_enhanced.json"

python -m src.benchmark.genaibench_enhance \
    --enhancer ${ENHANCER} \
    --backend ${BACKEND} \
    --enhancer_model_path ${ENHANCER_MODEL_PATH} \
    --input_file ${INPUT_FILE} \
    --output_file ${OUTPUT_FILE} \
    --temperature 0 \
    --top_p 1.0 \
    --max_new_tokens 512 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 4096