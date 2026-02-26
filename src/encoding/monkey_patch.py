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
    A wrapper for the original encode_prompt function.
    It checks if the prompt is a dictionary containing 'reasoning' and 'enhanced_prompt'.
    If so, it uses Method B (concatenating reasoning and prompt, then slicing).
    Otherwise, it falls back to the original behavior (Method A).
    """
    device = device or self._execution_device

    # Check if we are using Method B (passing a dict or list of dicts)
    is_method_b = False
    if isinstance(prompt, dict) and "reasoning" in prompt and "enhanced_prompt" in prompt:
        is_method_b = True
        prompts_list = [prompt]
    elif isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict) and "reasoning" in prompt[0] and "enhanced_prompt" in prompt[0]:
        is_method_b = True
        prompts_list = prompt
    else:
        # Method A: Standard behavior
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt_list) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt_list, device)

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

    # Method B Logic
    if is_method_b:
        batch_size = len(prompts_list)
        full_texts = []
        target_texts = []
        
        for p in prompts_list:
            reasoning = p["reasoning"]
            enhanced = p["enhanced_prompt"]
            # Concatenate reasoning and enhanced prompt. You can adjust the separator if needed.
            full_text = f"{reasoning}\n{enhanced}"
            full_texts.append(full_text)
            target_texts.append(enhanced)

        # 1. Encode the full concatenated text
        if prompt_embeds is None:
            full_prompt_embeds, full_prompt_embeds_mask = self._get_qwen_prompt_embeds(full_texts, device)
        else:
            full_prompt_embeds = prompt_embeds
            full_prompt_embeds_mask = prompt_embeds_mask

        # 2. Slice the hidden states to keep only the enhanced_prompt part
        # We need to import the utility function here to avoid circular imports if placed elsewhere, 
        # or assume it's available in the module scope.
        from .utils import slice_hidden_states
        
        sliced_embeds, sliced_masks = slice_hidden_states(
            self, full_texts, target_texts, full_prompt_embeds, full_prompt_embeds_mask, device, max_sequence_length
        )

        # 3. Apply the standard reshaping and repeating
        prompt_embeds = sliced_embeds[:, :max_sequence_length]
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if sliced_masks is not None:
            prompt_embeds_mask = sliced_masks[:, :max_sequence_length]
            prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
            prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

            if prompt_embeds_mask.all():
                prompt_embeds_mask = None

        return prompt_embeds, prompt_embeds_mask