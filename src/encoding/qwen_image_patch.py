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