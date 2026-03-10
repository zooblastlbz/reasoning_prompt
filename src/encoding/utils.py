import torch

# ---------------------------------------------------------------------------
# Section prefix constants
# Each section in the full text is annotated with a plain-text prefix so that
# the text encoder sees richer context structure.  The prefix tokens are then
# excluded when slicing out the target (enhanced_prompt) hidden states.
# ---------------------------------------------------------------------------
ORIGIN_PREFIX   = "[ORIGIN]: "
THINK_PREFIX    = "[THINK]: "
ENHANCED_PREFIX = "[ENHANCED]:"   # ← 去掉尾部空格，改用 \n 与 enhanced_prompt 分隔


def build_prefixed_full_text(origin_prompt: str, think: str, enhanced_prompt: str) -> tuple[str, int]:
    """
    Assemble the three-part full text that is fed to the text encoder.

    Format:
        [ORIGIN]: {origin_prompt}
        [THINK]: {think}
        [ENHANCED]:
        {enhanced_prompt}

    The section separator between [ENHANCED]: and enhanced_prompt is a newline
    character ('\\n') rather than a space.  This guarantees a clean BPE token
    boundary: tiktoken/Qwen always tokenises '\\n' as its own token, so
    char_start of enhanced_prompt will always coincide exactly with a token
    boundary in offset_mapping — no risk of the preceding space being fused
    into the first word of enhanced_prompt.

    Returns:
        (full_text, enhanced_char_start) — the concatenated string and the
        character-level start offset of enhanced_prompt within it.
    """
    prefix_part = (
        f"{ORIGIN_PREFIX}{origin_prompt}\n"
        f"{THINK_PREFIX}{think}\n"
        f"{ENHANCED_PREFIX}\n"   # ← \n 作为边界，enhanced_prompt 从下一行开始
    )
    full_text = prefix_part + enhanced_prompt
    enhanced_char_start = len(prefix_part)
    return full_text, enhanced_char_start


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
    enhanced_char_starts: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Slice hidden states that correspond to the enhanced_prompt tokens only,
    explicitly EXCLUDING the '[ENHANCED]: ' prefix tokens.

    Uses char_to_token mapping (from the tokenizer's encoding with
    return_offsets_mapping / offset_mapping) to convert the known character-level
    start position of enhanced_prompt into a token index.  This is robust to BPE
    context effects that break sub-sequence token matching.

    Args:
        enhanced_char_starts: Character-level start offset of enhanced_prompt
            inside each full_text (as returned by build_prefixed_full_text).
            If None, it is recomputed by searching for ENHANCED_PREFIX in the
            plain full_text string (fast string search, no tokenization needed).
    """
    template  = pipeline.prompt_template_encode
    drop_idx  = pipeline.prompt_template_encode_start_idx

    sliced_embeds_list = []

    for i in range(len(full_texts)):
        full_text = full_texts[i]

        # ── 1. Determine char-level start of enhanced_prompt ─────────────────
        if enhanced_char_starts is not None:
            char_start = enhanced_char_starts[i]
        else:
            # Fallback: plain string search — always correct regardless of BPE
            idx = full_text.rfind(ENHANCED_PREFIX)
            if idx == -1:
                print(
                    f"Warning: '[ENHANCED]: ' prefix not found in full_text[{i}] "
                    "(string search). Using all valid hidden states."
                )
                valid_len = (
                    int(prompt_embeds_mask[i].sum().item())
                    if prompt_embeds_mask is not None
                    else prompt_embeds.shape[1]
                )
                sliced_embed = prompt_embeds[i, :valid_len, :]
                sliced_embed = sliced_embed[:max_sequence_length]
                sliced_embeds_list.append(sliced_embed)
                continue
            char_start = idx + len(ENHANCED_PREFIX)

        # ── 2. Tokenize the template-wrapped full_text WITH offset_mapping ────
        wrapped = template.format(full_text)

        # Compute char offset introduced by the template prefix
        # (the template wraps the raw text, so we need to shift char_start)
        template_prefix_len = wrapped.find(full_text)
        if template_prefix_len == -1:
            # Unlikely, but guard: search for enhanced_prompt directly
            template_prefix_len = wrapped.find(enhanced_prompts[i])
            if template_prefix_len != -1:
                char_start_in_wrapped = template_prefix_len
            else:
                print(
                    f"Warning: could not locate full_text inside template-wrapped "
                    f"string for sample [{i}]. Using all valid hidden states."
                )
                valid_len = (
                    int(prompt_embeds_mask[i].sum().item())
                    if prompt_embeds_mask is not None
                    else prompt_embeds.shape[1]
                )
                sliced_embed = prompt_embeds[i, :valid_len, :]
                sliced_embed = sliced_embed[:max_sequence_length]
                sliced_embeds_list.append(sliced_embed)
                continue
        else:
            char_start_in_wrapped = template_prefix_len + char_start

        encoding = pipeline.tokenizer(
            wrapped,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offset_mapping = encoding["offset_mapping"]  # list of (char_start, char_end)

        # ── 3. Map char position → token index via offset_mapping ─────────────
        token_start_in_wrapped = None
        for tok_idx, (tok_char_start, tok_char_end) in enumerate(offset_mapping):
            if tok_char_start == char_start_in_wrapped:
                token_start_in_wrapped = tok_idx
                break
            # Handle the case where char_start_in_wrapped falls inside a token
            if tok_char_start < char_start_in_wrapped < tok_char_end:
                token_start_in_wrapped = tok_idx
                break

        if token_start_in_wrapped is None:
            print(
                f"Warning: char offset {char_start_in_wrapped} not found in "
                f"offset_mapping for sample [{i}]. Falling back to string search "
                f"on wrapped text."
            )
            # Last-resort: search for enhanced_prompt text in wrapped string
            char_start_in_wrapped = wrapped.rfind(enhanced_prompts[i])
            if char_start_in_wrapped == -1:
                valid_len = (
                    int(prompt_embeds_mask[i].sum().item())
                    if prompt_embeds_mask is not None
                    else prompt_embeds.shape[1]
                )
                sliced_embed = prompt_embeds[i, :valid_len, :]
                sliced_embed = sliced_embed[:max_sequence_length]
                sliced_embeds_list.append(sliced_embed)
                continue
            for tok_idx, (tok_char_start, tok_char_end) in enumerate(offset_mapping):
                if tok_char_start >= char_start_in_wrapped:
                    token_start_in_wrapped = tok_idx
                    break

        if token_start_in_wrapped is None:
            # Still not found — use all valid hidden states
            valid_len = (
                int(prompt_embeds_mask[i].sum().item())
                if prompt_embeds_mask is not None
                else prompt_embeds.shape[1]
            )
            sliced_embed = prompt_embeds[i, :valid_len, :]
            sliced_embed = sliced_embed[:max_sequence_length]
            sliced_embeds_list.append(sliced_embed)
            continue

        # ── 4. Compute length of enhanced_prompt in tokens ────────────────────
        enhanced_char_end = char_start_in_wrapped + len(enhanced_prompts[i])
        token_end_in_wrapped = len(offset_mapping)
        for tok_idx, (tok_char_start, _) in enumerate(offset_mapping):
            if tok_char_start >= enhanced_char_end:
                token_end_in_wrapped = tok_idx
                break
        target_len = token_end_in_wrapped - token_start_in_wrapped

        # ── 5. Convert to hidden-state index space (subtract drop_idx) ────────
        hs_start = token_start_in_wrapped - drop_idx
        sliced_embed = prompt_embeds[i, hs_start : hs_start + target_len, :]
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