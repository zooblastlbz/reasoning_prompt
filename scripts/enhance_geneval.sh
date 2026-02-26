#!/bin/bash
# GenEval Prompt Enhancement
# Usage: bash scripts/enhance_geneval.sh

ENHANCER="hunyuan"
ENHANCER_MODEL_PATH="/path/to/enhancer/model"
INPUT_FILE="data/prompts/geneval_metadata.jsonl"
OUTPUT_FILE="data/enhanced/geneval_enhanced.jsonl"

python -m src.benchmark.geneval_enhance \
    --enhancer ${ENHANCER} \
    --enhancer_model_path ${ENHANCER_MODEL_PATH} \
    --input_file ${INPUT_FILE} \
    --output_file ${OUTPUT_FILE}