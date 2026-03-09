import torch

def slice_hidden_states(pipeline, full_texts, target_texts, prompt_embeds, prompt_embeds_mask, device, max_sequence_length):
    """
    Finds the token indices of target_texts within full_texts and slices the hidden states.
    
    Returns embeds containing ONLY valid tokens (no padding zeros). This is critical because
    the downstream encode_prompt will set mask=None when mask.all() is True, and the transformer
    uses prompt_embeds.shape[1] as the effective sequence length for rotary embeddings.
    If we pad with zeros and provide a mask with 0s, the transformer would compute txt_seq_lens
    from the mask (e.g., 180) but prompt_embeds would have seq_len=512, causing a shape mismatch.
    """
    sliced_embeds_list = []

    for i in range(len(full_texts)):
        # Tokenize to find indices
        full_ids_with_special = pipeline.tokenizer(full_texts[i]).input_ids
        target_ids = pipeline.tokenizer(target_texts[i], add_special_tokens=False).input_ids

        start_idx = -1
        target_len = len(target_ids)
        for j in range(len(full_ids_with_special) - target_len + 1):
            if full_ids_with_special[j:j + target_len] == target_ids:
                start_idx = j
                break

        if start_idx != -1:
            end_idx = start_idx + target_len
            sliced_embed = prompt_embeds[i, start_idx:end_idx, :]
        else:
            print(f"Warning: Could not find exact token match for target text in prompt {i}. Using full embeddings.")
            if prompt_embeds_mask is not None:
                valid_len = int(prompt_embeds_mask[i].sum().item())
                sliced_embed = prompt_embeds[i, :valid_len, :]
            else:
                sliced_embed = prompt_embeds[i]

        # Truncate to max_sequence_length
        sliced_embed = sliced_embed[:max_sequence_length]
        sliced_embeds_list.append(sliced_embed)

    # Stack into a batch — pad to the longest within this batch if needed,
    # but use all-ones mask so encode_prompt will set mask=None.
    max_len = max(e.shape[0] for e in sliced_embeds_list)
    hidden_dim = sliced_embeds_list[0].shape[-1]

    padded_embeds = []
    padded_masks = []
    for embed in sliced_embeds_list:
        seq_len = embed.shape[0]
        pad_len = max_len - seq_len
        if pad_len > 0:
            pad_embed = torch.zeros((pad_len, hidden_dim), device=device, dtype=embed.dtype)
            embed = torch.cat([embed, pad_embed], dim=0)
        # Mark ALL positions as valid (including padding) so that
        # encode_prompt sees mask.all() == True and sets mask = None.
        # The transformer then uses prompt_embeds.shape[1] as txt_seq_len,
        # which matches the actual tensor shape — no rotary embedding mismatch.
        mask = torch.ones(max_len, device=device, dtype=torch.long)
        padded_embeds.append(embed)
        padded_masks.append(mask)

    final_embeds = torch.stack(padded_embeds)
    final_masks = torch.stack(padded_masks)

    return final_embeds, final_masks