#!/bin/bash
# GenEval Prompt Enhancement
# Usage: bash scripts/enhance_geneval.sh


BACKEND="vllm"
ENHANCER="hunyuan"
ENHANCER_MODEL_PATH="/ytech_m2v8_hdd/workspace/kling_mm/Models/Hunyuan-Prompt-Enhancer/reprompt/"
INPUT_FILE="data/prompts/geneval_metadata.jsonl"
OUTPUT_FILE="data/enhanced/geneval_enhanced.jsonl"


# vLLM specific
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=4096

python -m src.benchmark.geneval_enhance \
    --enhancer ${ENHANCER} \
    --backend ${BACKEND} \
    --enhancer_model_path ${ENHANCER_MODEL_PATH} \
    --input_file ${INPUT_FILE} \
    --output_file ${OUTPUT_FILE} \
    --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
    --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
    --max_model_len ${MAX_MODEL_LEN}