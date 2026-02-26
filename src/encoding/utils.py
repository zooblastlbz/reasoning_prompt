import torch

def slice_hidden_states(pipeline, full_texts, target_texts, prompt_embeds, prompt_embeds_mask, device, max_sequence_length):
    """
    Finds the token indices of target_texts within full_texts and slices the hidden states.
    """
    sliced_embeds_list = []
    sliced_masks_list = []
    
    for i in range(len(full_texts)):
        # Tokenize to find indices
        full_ids_with_special = pipeline.tokenizer(full_texts[i]).input_ids
        target_ids = pipeline.tokenizer(target_texts[i], add_special_tokens=False).input_ids
        
        start_idx = -1
        target_len = len(target_ids)
        for j in range(len(full_ids_with_special) - target_len + 1):
            if full_ids_with_special[j:j+target_len] == target_ids:
                start_idx = j
                break
                
        if start_idx != -1:
            end_idx = start_idx + target_len
            sliced_embed = prompt_embeds[i, start_idx:end_idx, :]
            if prompt_embeds_mask is not None:
                sliced_mask = prompt_embeds_mask[i, start_idx:end_idx]
            else:
                sliced_mask = torch.ones(target_len, device=device)
        else:
            print(f"Warning: Could not find exact token match for target text in prompt {i}. Using full embeddings.")
            sliced_embed = prompt_embeds[i]
            sliced_mask = prompt_embeds_mask[i] if prompt_embeds_mask is not None else torch.ones(sliced_embed.shape[0], device=device)
            
        # Pad back to max_sequence_length
        pad_len = max_sequence_length - sliced_embed.shape[0]
        if pad_len > 0:
            pad_embed = torch.zeros((pad_len, sliced_embed.shape[-1]), device=device, dtype=sliced_embed.dtype)
            sliced_embed = torch.cat([sliced_embed, pad_embed], dim=0)
            
            pad_mask = torch.zeros((pad_len,), device=device, dtype=sliced_mask.dtype)
            sliced_mask = torch.cat([sliced_mask, pad_mask], dim=0)
        else:
            sliced_embed = sliced_embed[:max_sequence_length]
            sliced_mask = sliced_mask[:max_sequence_length]
            
        sliced_embeds_list.append(sliced_embed)
        sliced_masks_list.append(sliced_mask)
        
    final_embeds = torch.stack(sliced_embeds_list)
    final_masks = torch.stack(sliced_masks_list)
    
    return final_embeds, final_masks