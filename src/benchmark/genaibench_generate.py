"""
GenAI-Bench Image Generation.

Generates images for the GenAI-Bench benchmark using a specified diffusion model,
with support for baseline and with_reasoning encoding methods.
Supports multi-GPU generation via accelerate PartialState.
All images are saved in a single flat directory with naming: {id}_sample_{sampleid}.jpeg

Usage (single GPU):
    python -m src.benchmark.genaibench_generate \
        --diffusion_model qwen_image \
        --model_path Qwen/Qwen-Image-2512 \
        --metadata_file data/enhanced/genaibench_enhanced.json \
        --output_dir data/generated_images/genaibench_results \
        --method baseline

Usage (multi-GPU):
    accelerate launch --num_processes NUM_GPUS \
        -m src.benchmark.genaibench_generate \
        --diffusion_model qwen_image \
        --model_path Qwen/Qwen-Image-2512 \
        --metadata_file data/enhanced/genaibench_enhanced.json \
        --output_dir data/generated_images/genaibench_results \
        --method baseline
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from accelerate import PartialState


NEGATIVE_PROMPT = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，"
    "人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
)


def load_pipeline(diffusion_model: str, model_path: str, torch_dtype, device):
    """
    Load the diffusion pipeline by model name.

    Args:
        diffusion_model: Name of the diffusion model. Supported: "qwen_image".
        model_path: Path or HuggingFace ID to the model weights.
        torch_dtype: Torch dtype for the model.
        device: Target device for this process.

    Returns:
        The loaded pipeline instance.
    """
    if diffusion_model == "qwen_image":
        from diffusers import QwenImagePipeline
        pipe = QwenImagePipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
    else:
        raise ValueError(
            f"Unknown diffusion model: '{diffusion_model}'. Supported: 'qwen_image'."
        )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Generate images for GenAI-Bench.")
    parser.add_argument("--diffusion_model", type=str, required=True,
                        help="Name of the diffusion model (e.g., 'qwen_image').")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path or HuggingFace ID of the diffusion model.")
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="Path to the enhanced GenAI-Bench metadata JSON file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base directory to save generated images.")
    parser.add_argument("--method", type=str, required=True,
                        choices=["baseline", "with_reasoning", "weighted_reasoning"],
                        help="Encoding method: 'baseline', 'with_reasoning', or 'weighted_reasoning'.")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number of images to generate per prompt.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--negative_prompt", type=str, default=NEGATIVE_PROMPT)
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Maximum sequence length for text encoding.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for reasoning-aware embeds in weighted_reasoning method. "
                             "1.0 = pure reasoning, 0.0 = pure plain. Default: 0.5.")
    args = parser.parse_args()

    # Setup distributed state
    distributed_state = PartialState()
    device = distributed_state.device
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[Process {distributed_state.process_index}] Loading {args.diffusion_model} "
          f"from {args.model_path} on {device}...")
    pipe = load_pipeline(args.diffusion_model, args.model_path, torch_dtype, device)

    # Load reasoning encoder if needed
    encode_fn = None
    if args.method in ("with_reasoning", "weighted_reasoning"):
        from src.encoding import get_encoder
        encode_fn = get_encoder(args.diffusion_model, method=args.method)

    # Read GenAI-Bench JSON (dict keyed by id)
    with open(args.metadata_file, "r", encoding="utf-8") as fp:
        all_data = json.load(fp)

    # Build ordered list of (id, entry) pairs
    items = [(entry_id, entry) for entry_id, entry in all_data.items()]

    # Output directory: output_dir/genaibench-{method}-{scale}-{steps}/
    run_name = f"genaibench-{args.method}-{int(args.guidance_scale)}-{args.num_inference_steps}"
    image_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(image_dir, exist_ok=True)

    # Split across GPUs using PartialState
    with distributed_state.split_between_processes(items) as local_items:
        for entry_id, entry in tqdm(
            local_items,
            desc=f"[GPU {distributed_state.process_index}] Generating",
            disable=not distributed_state.is_local_main_process,
        ):
            # Build pipeline call kwargs based on method
            if args.method == "baseline":
                prompt_kwargs = {
                    "prompt": entry.get("enhanced_prompt", entry.get("prompt", "")),
                }
            elif args.method == "with_reasoning":
                origin_prompt   = entry.get("original_prompt", entry.get("prompt", ""))
                think           = entry.get("reasoning", "")
                enhanced_prompt = entry.get("enhanced_prompt", entry.get("prompt", ""))
                prompt_embeds, prompt_embeds_mask = encode_fn(
                    pipe,
                    origin_prompt=origin_prompt,
                    think=think,
                    enhanced_prompt=enhanced_prompt,
                    device=device,
                    max_sequence_length=args.max_sequence_length,
                )
                prompt_kwargs = {
                    "prompt_embeds": prompt_embeds,
                    "prompt_embeds_mask": prompt_embeds_mask,
                }
            else:
                # weighted_reasoning
                origin_prompt   = entry.get("original_prompt", entry.get("prompt", ""))
                think           = entry.get("reasoning", "")
                enhanced_prompt = entry.get("enhanced_prompt", entry.get("prompt", ""))
                prompt_embeds, prompt_embeds_mask = encode_fn(
                    pipe,
                    origin_prompt=origin_prompt,
                    think=think,
                    enhanced_prompt=enhanced_prompt,
                    alpha=args.alpha,
                    device=device,
                    max_sequence_length=args.max_sequence_length,
                )
                prompt_kwargs = {
                    "prompt_embeds": prompt_embeds,
                    "prompt_embeds_mask": prompt_embeds_mask,
                }

            sample_count = 0
            for _ in range((args.n_samples + args.batch_size - 1) // args.batch_size):
                current_batch = min(args.batch_size, args.n_samples - sample_count)
                # Each sample gets a unique but reproducible seed
                sample_seed = args.seed + int(entry_id) * args.n_samples + sample_count
                generator = torch.Generator(device=device).manual_seed(sample_seed)

                with torch.autocast(str(device), dtype=torch_dtype):
                    images = pipe(
                        **prompt_kwargs,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                        true_cfg_scale=args.guidance_scale,
                        num_images_per_prompt=current_batch,
                        max_sequence_length=args.max_sequence_length,
                        generator=generator,
                    ).images

                for image in images:
                    # Naming: {id}_sample_{sampleid}.jpeg
                    filename = f"{entry_id}_sample_{sample_count}.jpeg"
                    image.save(os.path.join(image_dir, filename))
                    sample_count += 1

    # Wait for all processes to finish
    distributed_state.wait_for_everyone()
    if distributed_state.is_main_process:
        print(f"\nGeneration complete. Results saved to {image_dir}")


if __name__ == "__main__":
    main()
