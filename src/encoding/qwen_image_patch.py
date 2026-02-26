import json
import torch
from typing import Union, List, Optional


def encode_prompt_wrapper(
    self,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
    max_sequence_length: int = 1024,
):
    """
    A wrapper for the original encode_prompt function of QwenImagePipeline.
    
    Checks if the prompt string starts with "[WITH_REASONING]" prefix.
    If so, parses the JSON payload, encodes the full text (reasoning + enhanced prompt),
    and slices hidden states to keep only the enhanced prompt part.
    Otherwise, falls back to the original baseline behavior.
    The prefix itself does not participate in encoding.
    """
    device = device or self._execution_device

    is_with_reasoning = False
    prompts_list: list = []

    prompt_list_input = [prompt] if isinstance(prompt, str) else prompt

    for p in prompt_list_input:
        if isinstance(p, str) and p.startswith("[WITH_REASONING]"):
            is_with_reasoning = True
            try:
                parsed_p = json.loads(p[len("[WITH_REASONING]"):])
                prompts_list.append(parsed_p)
            except json.JSONDecodeError:
                print("Warning: Failed to parse JSON after [WITH_REASONING] prefix. Falling back to baseline.")
                is_with_reasoning = False
                prompts_list.append(p)
        else:
            prompts_list.append(p)

    if not is_with_reasoning:
        # Baseline: encode only the enhanced prompt string
        batch_size = len(prompts_list) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompts_list, device)

        assert prompt_embeds is not None
        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]
            prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
            prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)
            if prompt_embeds_mask.all():
                prompt_embeds_mask = None

        return prompt_embeds, prompt_embeds_mask

    # With Reasoning: encode full text, then slice out enhanced prompt hidden states
    batch_size = len(prompts_list)
    full_texts = []
    target_texts = []

    for p in prompts_list:
        if isinstance(p, dict) and "reasoning" in p and "enhanced_prompt" in p:
            full_text = f"{p['reasoning']}\n{p['enhanced_prompt']}"
            full_texts.append(full_text)
            target_texts.append(p["enhanced_prompt"])
        else:
            full_texts.append(str(p))
            target_texts.append(str(p))

    if prompt_embeds is None:
        full_prompt_embeds, full_prompt_embeds_mask = self._get_qwen_prompt_embeds(full_texts, device)
    else:
        full_prompt_embeds = prompt_embeds
        full_prompt_embeds_mask = prompt_embeds_mask

    from .utils import slice_hidden_states

    assert full_prompt_embeds is not None
    sliced_embeds, sliced_masks = slice_hidden_states(
        self, full_texts, target_texts,
        full_prompt_embeds, full_prompt_embeds_mask,
        device, max_sequence_length,
    )

    prompt_embeds = sliced_embeds[:, :max_sequence_length]
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    prompt_embeds_mask = sliced_masks[:, :max_sequence_length]
    prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
    prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)
    if prompt_embeds_mask.all():
        prompt_embeds_mask = None

    return prompt_embeds, prompt_embeds_mask