"""
GenAI-Bench Prompt Enhancement.

Reads GenAI-Bench metadata JSON, enhances prompts using a specified enhancer,
and saves the results while preserving all original fields.

Input format (JSON dict keyed by id):
    {
        "00000": {"id": "00000", "prompt": "...", ...},
        "00001": {"id": "00001", "prompt": "...", ...},
        ...
    }

Usage:
    python -m src.benchmark.genaibench_enhance \
        --enhancer hunyuan \
        --backend vllm \
        --enhancer_model_path /path/to/enhancer/model \
        --input_file data/prompts/genaibench_metadata.json \
        --output_file data/enhanced/genaibench_enhanced.json
"""

import json
import argparse
import os
from src.prompt_enhancement import get_enhancer, DEFAULT_SYS_PROMPT


def main():
    parser = argparse.ArgumentParser(description="Enhance GenAI-Bench prompts.")
    parser.add_argument("--enhancer", type=str, required=True,
                        help="Name of the enhancer to use (e.g., 'hunyuan').")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["transformers", "vllm"],
                        help="Inference backend: 'transformers' or 'vllm'.")
    parser.add_argument("--enhancer_model_path", type=str, required=True,
                        help="Path to the enhancer model.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the GenAI-Bench metadata JSON file.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the enhanced GenAI-Bench metadata.")
    parser.add_argument("--sys_prompt", type=str, default=DEFAULT_SYS_PROMPT,
                        help="System prompt for the enhancer.")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    # vLLM specific arguments
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (vllm backend only).")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization ratio (vllm backend only).")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Maximum model context length (vllm backend only).")
    args = parser.parse_args()

    # Build enhancer kwargs
    enhancer_kwargs = {
        "models_root_path": args.enhancer_model_path,
        "backend": args.backend,
    }
    if args.backend == "vllm":
        enhancer_kwargs.update({
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
        })

    enhancer = get_enhancer(args.enhancer, **enhancer_kwargs)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Read GenAI-Bench JSON (dict keyed by id)
    with open(args.input_file, "r", encoding="utf-8") as infile:
        all_data = json.load(infile)

    # Preserve key order
    ids = list(all_data.keys())
    entries = [all_data[k] for k in ids]
    original_prompts = [e.get("prompt", "") for e in entries]
    valid_indices = [i for i, p in enumerate(original_prompts) if p]

    if not valid_indices:
        print("No valid prompts found in input file.")
        return

    valid_prompts = [original_prompts[i] for i in valid_indices]
    print(f"Enhancing {len(valid_prompts)} prompts using {args.enhancer} ({args.backend} backend)...")

    # Batch inference
    results = enhancer.predict_batch(
        valid_prompts,
        sys_prompt=args.sys_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    # Write results — preserve original JSON dict format
    result_idx = 0
    for i in valid_indices:
        reasoning, enhanced, marked_output = results[result_idx]
        result_idx += 1

        original_prompt = entries[i]["prompt"]
        original_and_reasoning = (
            f"{original_prompt}\n{reasoning}" if reasoning else original_prompt
        )

        entries[i]["original_prompt"] = original_prompt
        entries[i]["reasoning"] = reasoning
        entries[i]["original_and_reasoning"] = original_and_reasoning
        entries[i]["enhanced_prompt"] = enhanced
        entries[i]["marked_output"] = marked_output

    output_data = {ids[i]: entries[i] for i in range(len(ids))}
    with open(args.output_file, "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, ensure_ascii=False, indent=4)

    print(f"Enhanced prompts saved to {args.output_file}")


if __name__ == "__main__":
    main()
