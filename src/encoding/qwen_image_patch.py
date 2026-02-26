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
    It checks if the prompt string starts with a specific prefix to determine the method.
    Prefix "[WITH_REASONING]" indicates using reasoning + enhanced prompt.
    Otherwise, it falls back to the original behavior (Method A).
    The prefix is removed before encoding.
    """
    device = device or self._execution_device

    # Check if we are using Method B based on the prefix
    is_method_b = False
    prompts_list = []
    
    prompt_list_input = [prompt] if isinstance(prompt, str) else prompt
    
    for p in prompt_list_input:
        if isinstance(p, str) and p.startswith("[WITH_REASONING]"):
            is_method_b = True
            # Remove the prefix and parse the JSON string back to a dict
            import json
            try:
                parsed_p = json.loads(p[len("[WITH_REASONING]"):])
                prompts_list.append(parsed_p)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse JSON after [WITH_REASONING] prefix. Falling back to Method A for this prompt.")
                is_method_b = False
                prompts_list.append(p)
        else:
            prompts_list.append(p)

    if not is_method_b:
        # Method A: Standard behavior
        batch_size = len(prompts_list) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompts_list, device)

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
            if isinstance(p, dict) and "reasoning" in p and "enhanced_prompt" in p:
                reasoning = p["reasoning"]
                enhanced = p["enhanced_prompt"]
                # Concatenate reasoning and enhanced prompt.
                full_text = f"{reasoning}\n{enhanced}"
                full_texts.append(full_text)
                target_texts.append(enhanced)
            else:
                # Fallback if parsing failed or format is wrong
                full_texts.append(str(p))
                target_texts.append(str(p))

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