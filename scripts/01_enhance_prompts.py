import json
import argparse
import os
from src.prompt_enhancement import HunyuanPromptEnhancer, DEFAULT_SYS_PROMPT

def main():
    parser = argparse.ArgumentParser(description="Enhance prompts using Hunyuan model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Hunyuan model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file containing prompts.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the enhanced prompts.")
    args = parser.parse_args()

    enhancer = HunyuanPromptEnhancer(models_root_path=args.model_path)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.input_file, 'r', encoding='utf-8') as infile, \
         open(args.output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            originalgit_prompt = data.get("prompt", "")
            if not original_prompt:
                continue
            
            print(f"Processing: {original_prompt}")
            reasoning, enhanced, marked_output = enhancer.predict(original_prompt, sys_prompt=DEFAULT_SYS_PROMPT)
            
            # 将原始输入和推理过程合并为一个字段，方便后续策略调用
            original_and_reasoning = f"{original_prompt}\n{reasoning}" if reasoning else original_prompt
            
            output_data = {
                "original_prompt": original_prompt,
                "reasoning": reasoning,
                "original_and_reasoning": original_and_reasoning,
                "enhanced_prompt": enhanced,
                "marked_output": marked_output
            }
            outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()