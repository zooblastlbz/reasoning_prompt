import json
import argparse
import os
import torch
from diffusers import QwenImagePipeline
from src.encoding import apply_monkey_patch

def main():
    parser = argparse.ArgumentParser(description="Generate images using the modified Qwen Image pipeline.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-2512", help="Path to the Qwen Image model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file containing enhanced prompts.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images.")
    parser.add_argument("--method", type=str, choices=["baseline", "with_reasoning"], required=True, help="Method to use: baseline or with_reasoning.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0, help="True CFG scale for Qwen Image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--aspect_ratio", type=str, default="16:9", choices=["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"], help="Aspect ratio for generation.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device and dtype
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16
        device = "cuda"
    else:
        torch_dtype = torch.float32
        device = "cpu"

    print(f"Loading model from {args.model_path} on {device}...")
    pipe = QwenImagePipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)

    # Apply monkey patch
    pipe = apply_monkey_patch(pipe)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1104),
        "3:4": (1104, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }
    width, height = aspect_ratios[args.aspect_ratio]

    negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"

    with open(args.input_file, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            data = json.loads(line)
            
            if args.method == "baseline":
                # Baseline: Pass only the enhanced prompt as a string
                prompt_input = data.get("enhanced_prompt", "")
            else:
                # With Reasoning: Pass a JSON string with the [WITH_REASONING] prefix
                # 使用合并后的 original_and_reasoning 字段
                prompt_dict = {
                    "reasoning": data.get("original_and_reasoning", ""),
                    "enhanced_prompt": data.get("enhanced_prompt", "")
                }
                prompt_input = f"[WITH_REASONING]{json.dumps(prompt_dict, ensure_ascii=False)}"

            print(f"Generating image {i} using method: {args.method}...")
            
            # The pipeline will call our patched encode_prompt internally
            image = pipe(
                prompt=prompt_input,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.true_cfg_scale,
                generator=generator
            ).images[0]

            output_path = os.path.join(args.output_dir, f"image_{i:04d}.png")
            image.save(output_path)
            print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()