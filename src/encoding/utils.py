import torch

# ---------------------------------------------------------------------------
# Section prefix constants
# Each section in the full text is annotated with a plain-text prefix so that
# the text encoder sees richer context structure.  The prefix tokens are then
# excluded when slicing out the target (enhanced_prompt) hidden states.
# ---------------------------------------------------------------------------
ORIGIN_PREFIX   = "[ORIGIN]: "
THINK_PREFIX    = "[THINK]: "
ENHANCED_PREFIX = "[ENHANCED]: "


def build_prefixed_full_text(origin_prompt: str, think: str, enhanced_prompt: str) -> str:
    """
    Assemble the three-part full text that is fed to the text encoder.

    Format:
        [ORIGIN]: {origin_prompt}
        [THINK]: {think}
        [ENHANCED]: {enhanced_prompt}

    Returns:
        The concatenated string.
    """
    return (
        f"{ORIGIN_PREFIX}{origin_prompt}\n"
        f"{THINK_PREFIX}{think}\n"
        f"{ENHANCED_PREFIX}{enhanced_prompt}"
    )


def get_qwen_prompt_embeds_no_limit(
    pipeline,
    texts: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reproduce the logic of QwenImagePipeline._get_qwen_prompt_embeds but WITHOUT
    the tokenizer max_length cap (i.e. no truncation).

    The original function calls:
        self.tokenizer(txt, max_length=self.tokenizer_max_length + drop_idx,
                       padding=True, truncation=True, ...)
    which hard-clips the full text to tokenizer_max_length tokens.
    Here we set truncation=False so the reasoning context is never cut off.

    Args:
        pipeline: QwenImagePipeline instance.
        texts:    List of raw prompt strings (before template wrapping).
        device:   Target device.

    Returns:
        (prompt_embeds, encoder_attention_mask) — same shapes/semantics as the
        original function.
    """
    dtype = pipeline.text_encoder.dtype
    template  = pipeline.prompt_template_encode
    drop_idx  = pipeline.prompt_template_encode_start_idx

    # Wrap each text in the pipeline's own prompt template
    txt = [template.format(e) for e in texts]

    # Tokenize WITHOUT truncation so long reasoning texts are preserved in full
    txt_tokens = pipeline.tokenizer(
        txt,
        padding=True,
        truncation=False,          # ← key difference from the original
        return_tensors="pt",
    ).to(device)

    encoder_hidden_states = pipeline.text_encoder(
        input_ids=txt_tokens.input_ids,
        attention_mask=txt_tokens.attention_mask,
        output_hidden_states=True,
    )
    hidden_states = encoder_hidden_states.hidden_states[-1]

    # Use the pipeline's own helper to strip padding hidden states per sample
    split_hidden_states = pipeline._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)

    # Drop the template prefix tokens (same as original)
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]

    attn_mask_list = [
        torch.ones(e.size(0), dtype=torch.long, device=e.device)
        for e in split_hidden_states
    ]

    max_seq_len = max(e.size(0) for e in split_hidden_states)
    prompt_embeds = torch.stack([
        torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
        for u in split_hidden_states
    ])
    encoder_attention_mask = torch.stack([
        torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
        for u in attn_mask_list
    ])

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    return prompt_embeds, encoder_attention_mask


def slice_hidden_states_by_prefix(
    pipeline,
    full_texts: list[str],
    enhanced_prompts: list[str],
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    device: torch.device,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Slice hidden states that correspond to the enhanced_prompt tokens only,
    explicitly EXCLUDING the '[ENHANCED]: ' prefix tokens.

    Strategy
    --------
    For each sample we tokenize:
      1. full_text  → find the absolute position of the ENHANCED_PREFIX inside it
      2. prefix     → count how many tokens the prefix occupies
      3. target     → the enhanced_prompt tokens (without special tokens)

    The slice window is [prefix_end_idx : prefix_end_idx + target_len].

    This guarantees we never include the '[ENHANCED]: ' tokens in the final
    embeddings that are passed to the diffusion model.

    Returns
    -------
    (final_embeds, final_masks) — same format as slice_hidden_states.
    """
    # Tokenize the prefix once (it is the same for every sample)
    prefix_ids = pipeline.tokenizer(
        ENHANCED_PREFIX, add_special_tokens=False
    ).input_ids
    prefix_len = len(prefix_ids)

    sliced_embeds_list = []

    for i in range(len(full_texts)):
        full_ids = pipeline.tokenizer(full_texts[i]).input_ids
        target_ids = pipeline.tokenizer(
            enhanced_prompts[i], add_special_tokens=False
        ).input_ids
        target_len = len(target_ids)

        # ── Step 1: find '[ENHANCED]: ' inside full_ids ──────────────────────
        prefix_start = -1
        for j in range(len(full_ids) - prefix_len + 1):
            if full_ids[j : j + prefix_len] == prefix_ids:
                prefix_start = j
                # Take the LAST occurrence (in case the prefix string appears
                # earlier by coincidence)
                # We intentionally keep iterating to get the last match.
        # Use last match: search from end
        for j in range(len(full_ids) - prefix_len, -1, -1):
            if full_ids[j : j + prefix_len] == prefix_ids:
                prefix_start = j
                break

        if prefix_start == -1:
            # Fallback: prefix not found — use the plain subseq-match strategy
            print(
                f"Warning: '[ENHANCED]: ' prefix not found in full_text[{i}]. "
                "Falling back to plain token match."
            )
            start_idx = -1
            for j in range(len(full_ids) - target_len + 1):
                if full_ids[j : j + target_len] == target_ids:
                    start_idx = j
                    break
            if start_idx == -1:
                print(
                    f"Warning: enhanced_prompt tokens not found in full_text[{i}]. "
                    "Using all valid hidden states."
                )
                valid_len = int(prompt_embeds_mask[i].sum().item()) if prompt_embeds_mask is not None else prompt_embeds.shape[1]
                sliced_embed = prompt_embeds[i, :valid_len, :]
            else:
                sliced_embed = prompt_embeds[i, start_idx : start_idx + target_len, :]
        else:
            # ── Step 2: the enhanced_prompt starts right after the prefix ────
            enhanced_start = prefix_start + prefix_len
            sliced_embed = prompt_embeds[i, enhanced_start : enhanced_start + target_len, :]

        # Truncate to max_sequence_length
        sliced_embed = sliced_embed[:max_sequence_length]
        sliced_embeds_list.append(sliced_embed)

    # ── Pad batch to uniform length, all-ones mask ───────────────────────────
    max_len    = max(e.shape[0] for e in sliced_embeds_list)
    hidden_dim = sliced_embeds_list[0].shape[-1]

    padded_embeds = []
    padded_masks  = []
    for embed in sliced_embeds_list:
        pad_len = max_len - embed.shape[0]
        if pad_len > 0:
            pad = torch.zeros((pad_len, hidden_dim), device=device, dtype=embed.dtype)
            embed = torch.cat([embed, pad], dim=0)
        padded_embeds.append(embed)
        padded_masks.append(torch.ones(max_len, device=device, dtype=torch.long))

    return torch.stack(padded_embeds), torch.stack(padded_masks)


# ---------------------------------------------------------------------------
# Original slice_hidden_states kept for backward compatibility
# ---------------------------------------------------------------------------
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