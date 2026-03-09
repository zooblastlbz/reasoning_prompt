"""
GenEval Image Generation.

Generates images for the GenEval benchmark using a specified diffusion model,
with support for baseline and with_reasoning encoding methods.
Supports multi-GPU generation via accelerate PartialState.
Output follows the GenEval directory structure for direct evaluation.

Usage (single GPU):
    python -m src.benchmark.geneval_generate \
        --diffusion_model qwen_image \
        --model_path Qwen/Qwen-Image-2512 \
        --metadata_file data/enhanced/geneval_enhanced.jsonl \
        --output_dir data/generated_images/geneval_results \
        --method baseline

Usage (multi-GPU):
    accelerate launch --num_processes NUM_GPUS \
        -m src.benchmark.geneval_generate \
        --diffusion_model qwen_image \
        --model_path Qwen/Qwen-Image-2512 \
        --metadata_file data/enhanced/geneval_enhanced.jsonl \
        --output_dir data/generated_images/geneval_results \
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
        The loaded pipeline instance (already patched).
    """
    from src.encoding import apply_patch

    if diffusion_model == "qwen_image":
        from diffusers import QwenImagePipeline
        pipe = QwenImagePipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
    else:
        raise ValueError(
            f"Unknown diffusion model: '{diffusion_model}'. Supported: 'qwen_image'."
        )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe = apply_patch(pipe, model_type=diffusion_model)
    return pipe


def build_prompt_input(metadata: dict, method: str) -> str:
    """
    Build the prompt string to pass to the pipeline based on the method.

    Args:
        metadata: A single GenEval metadata entry (with enhanced fields).
        method: "baseline" or "with_reasoning".

    Returns:
        The prompt string (with prefix for with_reasoning).
    """
    if method == "baseline":
        return metadata.get("enhanced_prompt", metadata.get("prompt", ""))
    elif method == "with_reasoning":
        prompt_dict = {
            "reasoning": metadata.get("original_and_reasoning", ""),
            "enhanced_prompt": metadata.get("enhanced_prompt", metadata.get("prompt", "")),
        }
        return f"[WITH_REASONING]{json.dumps(prompt_dict, ensure_ascii=False)}"
    else:
        raise ValueError(f"Unknown method: '{method}'. Supported: 'baseline', 'with_reasoning'.")


def main():
    parser = argparse.ArgumentParser(description="Generate images for GenEval.")
    parser.add_argument("--diffusion_model", type=str, required=True,
                        help="Name of the diffusion model (e.g., 'qwen_image').")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path or HuggingFace ID of the diffusion model.")
    parser.add_argument("--metadata_file", type=str, required=True,
                        help="Path to the enhanced GenEval metadata JSONL file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base directory to save generated images.")
    parser.add_argument("--method", type=str, required=True,
                        choices=["baseline", "with_reasoning"],
                        help="Encoding method: 'baseline' or 'with_reasoning'.")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number of images to generate per prompt.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--negative_prompt", type=str, default=NEGATIVE_PROMPT)
    args = parser.parse_args()

    # Setup distributed state
    distributed_state = PartialState()
    device = distributed_state.device
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[Process {distributed_state.process_index}] Loading {args.diffusion_model} "
          f"from {args.model_path} on {device}...")
    pipe = load_pipeline(args.diffusion_model, args.model_path, torch_dtype, device)

    with open(args.metadata_file, "r", encoding="utf-8") as fp:
        metadatas = [json.loads(line) for line in fp]

    # GenEval directory structure:
    # output_dir/geneval-{method}-{scale}-{steps}/{index:05d}/samples/{sample:05d}.png
    run_name = f"geneval-{args.method}-{int(args.guidance_scale)}-{args.num_inference_steps}"
    base_outpath = os.path.join(args.output_dir, run_name)
    os.makedirs(base_outpath, exist_ok=True)

    # Split samples across GPUs using PartialState
    indexed_metadatas = list(enumerate(metadatas))
    with distributed_state.split_between_processes(indexed_metadatas) as local_samples:
        for index, metadata in tqdm(
            local_samples,
            desc=f"[GPU {distributed_state.process_index}] Generating",
            disable=not distributed_state.is_local_main_process,
        ):
            outpath = os.path.join(base_outpath, f"{index:0>5}")
            sample_path = os.path.join(outpath, "samples")
            os.makedirs(sample_path, exist_ok=True)

            # Save per-prompt metadata (required by GenEval evaluation)
            with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
                json.dump(metadata, fp, ensure_ascii=False)

            prompt_input = build_prompt_input(metadata, args.method)

            sample_count = 0
            for _ in range((args.n_samples + args.batch_size - 1) // args.batch_size):
                current_batch = min(args.batch_size, args.n_samples - sample_count)
                generator = torch.Generator(device=device).manual_seed(args.seed)

                with torch.autocast(str(device), dtype=torch_dtype):
                    images = pipe(
                        prompt=prompt_input,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                        true_cfg_scale=args.guidance_scale,
                        num_images_per_prompt=current_batch,
                        generator=generator,
                    ).images

                for image in images:
                    image.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                    sample_count += 1

    # Wait for all processes to finish
    distributed_state.wait_for_everyone()
    if distributed_state.is_main_process:
        print(f"\nGeneration complete. Results saved to {base_outpath}")


if __name__ == "__main__":
    main()