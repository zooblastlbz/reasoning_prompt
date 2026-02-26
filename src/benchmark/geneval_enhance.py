"""
GenEval Prompt Enhancement.

Reads GenEval metadata JSONL, enhances prompts using a specified enhancer,
and saves the results while preserving all original GenEval fields.

Usage:
    python -m src.benchmark.geneval_enhance \
        --enhancer hunyuan \
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
    args = parser.parse_args()

    enhancer = get_enhancer(args.enhancer, models_root_path=args.enhancer_model_path)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.input_file, "r", encoding="utf-8") as infile, \
         open(args.output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            metadata = json.loads(line)
            original_prompt = metadata.get("prompt", "")
            if not original_prompt:
                continue

            print(f"Processing: {original_prompt}")
            reasoning, enhanced, marked_output = enhancer.predict(
                original_prompt,
                sys_prompt=args.sys_prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )

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