import json
import torch
from typing import Union, List, Optional
from .utils import (
    build_prefixed_full_text,
    get_qwen_prompt_embeds_no_limit,
    slice_hidden_states_by_prefix,
    slice_hidden_states,
)


def encode_with_reasoning(
    pipeline,
    origin_prompt: str,
    think: str,
    enhanced_prompt: str,
    device: Optional[torch.device] = None,
    max_sequence_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a three-part full text (origin_prompt / think / enhanced_prompt) using
    the pipeline's text encoder WITHOUT any token-length cap, then slice out only
    the hidden states that correspond to the enhanced_prompt tokens.

    Each section is annotated with a plain-text prefix:
        [ORIGIN]: {origin_prompt}
        [THINK]:  {think}
        [ENHANCED]: {enhanced_prompt}

    The prefix tokens of [ENHANCED] are excluded from the returned embeddings.

    Args:
        pipeline:             QwenImagePipeline instance.
        origin_prompt:        The original user prompt.
        think:                The reasoning text produced by the enhancer.
        enhanced_prompt:      The final enhanced prompt to generate from.
        device:               Target device.
        max_sequence_length:  Max tokens kept from the enhanced_prompt slice.

    Returns:
        (prompt_embeds, prompt_embeds_mask) ready for pipeline.__call__.
    """
    device = device or pipeline._execution_device

    # 1. Build three-part prefixed text — also get char-level start of enhanced_prompt
    full_text, enhanced_char_start = build_prefixed_full_text(origin_prompt, think, enhanced_prompt)

    # 2. Encode without length limit
    full_prompt_embeds, full_prompt_embeds_mask = get_qwen_prompt_embeds_no_limit(
        pipeline, [full_text], device
    )

    # 3. Slice out enhanced_prompt hidden states using char offset (BPE-safe)
    prompt_embeds, prompt_embeds_mask = slice_hidden_states_by_prefix(
        pipeline, [full_text], [enhanced_prompt],
        full_prompt_embeds, full_prompt_embeds_mask,
        device, max_sequence_length,
        enhanced_char_starts=[enhanced_char_start],
    )

    return prompt_embeds, prompt_embeds_mask


def encode_with_weighted_reasoning(
    pipeline,
    origin_prompt: str,
    think: str,
    enhanced_prompt: str,
    alpha: float = 0.5,
    device: Optional[torch.device] = None,
    max_sequence_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted combination of reasoning-aware and reasoning-free hidden states.

    Computes:
        final_embeds = alpha * reasoning_embeds + (1 - alpha) * plain_embeds

    where reasoning_embeds are the enhanced_prompt hidden states sliced from the
    three-part prefixed encoding, and plain_embeds are the hidden states from
    encoding the enhanced_prompt alone (standard pipeline call).

    Args:
        pipeline:             QwenImagePipeline instance.
        origin_prompt:        The original user prompt.
        think:                The reasoning text produced by the enhancer.
        enhanced_prompt:      The final enhanced prompt.
        alpha:                Weight for reasoning-aware embeds.
                              1.0 = pure reasoning, 0.0 = pure plain.
        device:               Target device.
        max_sequence_length:  Max tokens for the output embeddings.

    Returns:
        (prompt_embeds, prompt_embeds_mask) ready for pipeline.__call__.
    """
    device = device or pipeline._execution_device

    # 1. Reasoning-aware embeds (three-part encoding → slice enhanced_prompt)
    reasoning_embeds, _ = encode_with_reasoning(
        pipeline, origin_prompt, think, enhanced_prompt, device, max_sequence_length,
    )

    # 2. Plain embeds (encode enhanced_prompt alone, original pipeline function)
    plain_embeds, _ = pipeline._get_qwen_prompt_embeds([enhanced_prompt], device)
    plain_embeds = plain_embeds[:, :max_sequence_length]

    # 3. Align sequence lengths
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

    # 5. All-ones mask
    seq_len = final_embeds.shape[1]
    final_mask = torch.ones((1, seq_len), device=device, dtype=torch.long)

    return final_embeds, final_mask


def batch_encode_with_reasoning(
    pipeline,
    origin_prompts: list[str],
    thinks: list[str],
    enhanced_prompts: list[str],
    device: Optional[torch.device] = None,
    max_sequence_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batch version of encode_with_reasoning.

    Args:
        pipeline:          QwenImagePipeline instance.
        origin_prompts:    List of original user prompts.
        thinks:            List of reasoning texts.
        enhanced_prompts:  List of enhanced prompts.
        device:            Target device.
        max_sequence_length: Max tokens for the output embeddings.

    Returns:
        (prompt_embeds, prompt_embeds_mask) ready for pipeline.__call__.
    """
    device = device or pipeline._execution_device

    # Build texts and collect char-level offsets in one pass
    full_texts = []
    enhanced_char_starts = []
    for o, t, e in zip(origin_prompts, thinks, enhanced_prompts):
        full_text, char_start = build_prefixed_full_text(o, t, e)
        full_texts.append(full_text)
        enhanced_char_starts.append(char_start)

    full_prompt_embeds, full_prompt_embeds_mask = get_qwen_prompt_embeds_no_limit(
        pipeline, full_texts, device
    )

    prompt_embeds, prompt_embeds_mask = slice_hidden_states_by_prefix(
        pipeline, full_texts, enhanced_prompts,
        full_prompt_embeds, full_prompt_embeds_mask,
        device, max_sequence_length,
        enhanced_char_starts=enhanced_char_starts,
    )

    return prompt_embeds, prompt_embeds_mask