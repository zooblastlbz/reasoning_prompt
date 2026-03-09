import json
import torch
from typing import Union, List, Optional
from .utils import slice_hidden_states


def encode_with_reasoning(
    pipeline,
    reasoning_text: str,
    enhanced_prompt: str,
    device: Optional[torch.device] = None,
    max_sequence_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode the full text (reasoning + enhanced prompt) using the pipeline's text encoder,
    then slice out only the hidden states corresponding to the enhanced prompt.

    Args:
        pipeline: The QwenImagePipeline instance (unpatched).
        reasoning_text: The reasoning context (original_prompt + reasoning).
        enhanced_prompt: The final enhanced prompt.
        device: Target device.
        max_sequence_length: Maximum sequence length for the output embeddings.

    Returns:
        tuple: (prompt_embeds, prompt_embeds_mask) ready to pass to pipeline.__call__.
    """
    device = device or pipeline._execution_device

    # 1. Concatenate reasoning and enhanced prompt as a single text
    full_text = f"{reasoning_text}\n{enhanced_prompt}"

    # 2. Encode the full concatenated text using the pipeline's own text encoder
    full_prompt_embeds, full_prompt_embeds_mask = pipeline._get_qwen_prompt_embeds(
        [full_text], device
    )

    # 3. Slice out the hidden states corresponding to the enhanced prompt only
    prompt_embeds, prompt_embeds_mask = slice_hidden_states(
        pipeline, [full_text], [enhanced_prompt],
        full_prompt_embeds, full_prompt_embeds_mask,
        device, max_sequence_length,
    )

    return prompt_embeds, prompt_embeds_mask


def encode_with_weighted_reasoning(
    pipeline,
    reasoning_text: str,
    enhanced_prompt: str,
    alpha: float = 0.5,
    device: Optional[torch.device] = None,
    max_sequence_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted combination of reasoning-aware and reasoning-free hidden states.

    Computes:
        final_embeds = alpha * reasoning_embeds + (1 - alpha) * plain_embeds

    where reasoning_embeds are the enhanced_prompt hidden states extracted from
    encoding the full (reasoning + enhanced_prompt) text, and plain_embeds are
    the hidden states from encoding enhanced_prompt alone.

    Args:
        pipeline: The QwenImagePipeline instance.
        reasoning_text: The reasoning context (original_prompt + reasoning).
        enhanced_prompt: The final enhanced prompt.
        alpha: Weight for reasoning-aware embeds. 1.0 = pure reasoning,
               0.0 = pure plain encoding. Default: 0.5.
        device: Target device.
        max_sequence_length: Maximum sequence length for the output embeddings.

    Returns:
        tuple: (prompt_embeds, prompt_embeds_mask) ready to pass to pipeline.__call__.
    """
    device = device or pipeline._execution_device

    # 1. Encode WITH reasoning context, slice out enhanced_prompt hidden states
    reasoning_embeds, reasoning_mask = encode_with_reasoning(
        pipeline, reasoning_text, enhanced_prompt, device, max_sequence_length,
    )

    # 2. Encode WITHOUT reasoning context (plain enhanced_prompt only)
    plain_embeds, plain_mask = pipeline._get_qwen_prompt_embeds(
        [enhanced_prompt], device
    )
    # Truncate to max_sequence_length to match downstream expectations
    plain_embeds = plain_embeds[:, :max_sequence_length]

    # 3. Align sequence lengths — pad the shorter one to match the longer
    r_len = reasoning_embeds.shape[1]
    p_len = plain_embeds.shape[1]
    hidden_dim = reasoning_embeds.shape[2]

    if r_len < p_len:
        pad = torch.zeros((1, p_len - r_len, hidden_dim), device=device, dtype=reasoning_embeds.dtype)
        reasoning_embeds = torch.cat([reasoning_embeds, pad], dim=1)
    elif p_len < r_len:
        pad = torch.zeros((1, r_len - p_len, hidden_dim), device=device, dtype=plain_embeds.dtype)
        plain_embeds = torch.cat([plain_embeds, pad], dim=1)

    # 4. Weighted combination
    final_embeds = alpha * reasoning_embeds + (1 - alpha) * plain_embeds

    # 5. All-ones mask to ensure encode_prompt sets mask=None
    seq_len = final_embeds.shape[1]
    final_mask = torch.ones((1, seq_len), device=device, dtype=torch.long)

    return final_embeds, final_mask


def batch_encode_with_reasoning(
    pipeline,
    reasoning_texts: list[str],
    enhanced_prompts: list[str],
    device: Optional[torch.device] = None,
    max_sequence_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batch version of encode_with_reasoning.

    Args:
        pipeline: The QwenImagePipeline instance (unpatched).
        reasoning_texts: List of reasoning contexts.
        enhanced_prompts: List of enhanced prompts.
        device: Target device.
        max_sequence_length: Maximum sequence length for the output embeddings.

    Returns:
        tuple: (prompt_embeds, prompt_embeds_mask) ready to pass to pipeline.__call__.
    """
    device = device or pipeline._execution_device

    full_texts = [
        f"{r}\n{e}" for r, e in zip(reasoning_texts, enhanced_prompts)
    ]

    full_prompt_embeds, full_prompt_embeds_mask = pipeline._get_qwen_prompt_embeds(
        full_texts, device
    )

    prompt_embeds, prompt_embeds_mask = slice_hidden_states(
        pipeline, full_texts, enhanced_prompts,
        full_prompt_embeds, full_prompt_embeds_mask,
        device, max_sequence_length,
    )

    return prompt_embeds, prompt_embeds_mask