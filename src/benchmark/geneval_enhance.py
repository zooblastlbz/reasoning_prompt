"""
GenEval Prompt Enhancement.

Reads GenEval metadata JSONL, enhances prompts using a specified enhancer,
and saves the results while preserving all original GenEval fields.

Usage:
    python -m src.benchmark.geneval_enhance \
        --enhancer hunyuan \
        --backend vllm \
        --enhancer_model_path /path/to/enhancer/model \
        --input_file data/prompts/geneval_metadata.jsonl \
        --output_file data/enhanced/geneval_enhanced.jsonl
"""

import json
import argparse
import os
from src.prompt_enhancement import get_enhancer, DEFAULT_SYS_PROMPT


def main():
    parser = argparse.ArgumentParser(description="Enhance GenEval prompts.")
    parser.add_argument("--enhancer", type=str, required=True,
                        help="Name of the enhancer to use (e.g., 'hunyuan').")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["transformers", "vllm"],
                        help="Inference backend: 'transformers' or 'vllm'.")
    parser.add_argument("--enhancer_model_path", type=str, required=True,
                        help="Path to the enhancer model.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the GenEval metadata JSONL file.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the enhanced GenEval metadata.")
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

    # Read all metadata
    with open(args.input_file, "r", encoding="utf-8") as infile:
        all_metadatas = [json.loads(line) for line in infile]

    original_prompts = [m.get("prompt", "") for m in all_metadatas]
    valid_indices = [i for i, p in enumerate(original_prompts) if p]

    if not valid_indices:
        print("No valid prompts found in input file.")
        return

    valid_prompts = [original_prompts[i] for i in valid_indices]
    print(f"Enhancing {len(valid_prompts)} prompts using {args.enhancer} ({args.backend} backend)...")

    # Batch inference — vLLM processes all prompts in parallel,
    # transformers falls back to sequential
    results = enhancer.predict_batch(
        valid_prompts,
        sys_prompt=args.sys_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )

    # Write results
    with open(args.output_file, "w", encoding="utf-8") as outfile:
        result_idx = 0
        for i, metadata in enumerate(all_metadatas):
            if i not in valid_indices:
                continue

            reasoning, enhanced, marked_output = results[result_idx]
            result_idx += 1

            original_prompt = metadata["prompt"]
            original_and_reasoning = (
                f"{original_prompt}\n{reasoning}" if reasoning else original_prompt
            )

            metadata["original_prompt"] = original_prompt
            metadata["reasoning"] = reasoning
            metadata["original_and_reasoning"] = original_and_reasoning
            metadata["enhanced_prompt"] = enhanced
            metadata["marked_output"] = marked_output

            outfile.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    print(f"Enhanced prompts saved to {args.output_file}")


if __name__ == "__main__":
    main()