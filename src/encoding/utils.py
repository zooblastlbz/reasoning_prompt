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


def split_prompt_template(template: str) -> tuple[str, str]:
    """
    Split a template like "...{}..." into (prefix, suffix).

    The Qwen image prompt template is expected to contain exactly one "{}"
    placeholder where user text is inserted.
    """
    if template.count("{}") != 1:
        raise ValueError(
            f"prompt_template_encode must contain exactly one '{{}}', got {template.count('{}')}"
        )
    return template.split("{}", 1)


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
    include_template_suffix: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Slice hidden states that correspond to the enhanced_prompt tokens,
    excluding the '[ENHANCED]:' prefix tokens.

    Optionally include template suffix tokens in the sliced length
    (enhanced_len + template_suffix_len) to align with plain
    _get_qwen_prompt_embeds semantics.

    Strategy (no return_offsets_mapping needed, works with slow tokenizers):
    -------------------------------------------------------------------------
    Because build_prefixed_full_text separates sections with '\\n' — a token
    that tiktoken/BPE *always* splits independently — we can safely tokenize
    the prefix part and the enhanced_prompt part *separately* and trust that
    their token counts add up correctly inside the full tokenization.

    Concretely, for the template-wrapped text:
        wrapped = template.format(full_text)
               = <template_prefix> + full_text
               = <template_prefix> + prefix_part + enhanced_prompt

    We tokenize <template_prefix> + prefix_part together (without the
    enhanced_prompt), count the resulting tokens, subtract drop_idx to get
    hs_start, then tokenize enhanced_prompt alone to get target_len.

    This is valid because:
      1. The trailing '\\n' of prefix_part is always its own token — it never
         merges with the first token of enhanced_prompt.
      2. drop_idx is the number of template-prefix tokens the pipeline strips.
    """
    template = pipeline.prompt_template_encode
    drop_idx = pipeline.prompt_template_encode_start_idx
    template_prefix, template_suffix = split_prompt_template(template)

    template_suffix_token_len = 0
    if include_template_suffix:
        template_suffix_token_len = len(
            pipeline.tokenizer(template_suffix, add_special_tokens=False).input_ids
        )

    sliced_embeds_list = []

    for i in range(len(full_texts)):
        full_text = full_texts[i]
        enhanced_prompt = enhanced_prompts[i]

        # ── 1. Reconstruct prefix_part (everything before enhanced_prompt) ───
        if enhanced_char_starts is not None:
            # char start is known — slice directly
            prefix_part = full_text[: enhanced_char_starts[i]]
        else:
            # Fallback: find via string search (always correct, no tokenizer)
            idx = full_text.rfind(ENHANCED_PREFIX)
            if idx == -1:
                print(
                    f"Warning: ENHANCED_PREFIX not found in full_text[{i}]. "
                    "Using all valid hidden states."
                )
                valid_len = (
                    int(prompt_embeds_mask[i].sum().item())
                    if prompt_embeds_mask is not None
                    else prompt_embeds.shape[1]
                )
                sliced_embeds_list.append(prompt_embeds[i, :valid_len, :])
                continue
            # prefix_part ends right after the trailing \n of ENHANCED_PREFIX\n
            char_start = idx + len(ENHANCED_PREFIX) + 1  # +1 for the \n separator
            prefix_part = full_text[:char_start]

        # ── 2. Build prefix-only wrapped text (exclude template suffix) ───────
        # We need token position where enhanced_prompt starts inside:
        #   template_prefix + prefix_part + enhanced_prompt + template_suffix
        # So the counting text here must be:
        #   template_prefix + prefix_part
        wrapped_prefix = f"{template_prefix}{prefix_part}"

        # ── 3. Count tokens in wrapped_prefix → tells us where enhanced_prompt
        #       starts in the full token sequence ─────────────────────────────
        prefix_token_count = len(
            pipeline.tokenizer(wrapped_prefix, add_special_tokens=False).input_ids
        )

        # ── 4. Count tokens in enhanced_prompt (+ optional template suffix) ───
        # Safe because the leading \n boundary means BPE will produce the same
        # tokens for enhanced_prompt regardless of surrounding context.
        target_ids = pipeline.tokenizer(
            enhanced_prompt, add_special_tokens=False
        ).input_ids
        target_len = len(target_ids) + template_suffix_token_len

        # ── 5. Convert to hidden-state index (subtract drop_idx) ─────────────
        hs_start = prefix_token_count - drop_idx
        if hs_start < 0:
            print(
                f"Warning: hs_start={hs_start} < 0 for sample [{i}] "
                "(prefix shorter than drop_idx?). Using all valid hidden states."
            )
            valid_len = (
                int(prompt_embeds_mask[i].sum().item())
                if prompt_embeds_mask is not None
                else prompt_embeds.shape[1]
            )
            sliced_embeds_list.append(prompt_embeds[i, :valid_len, :])
            continue

        sliced_embed = prompt_embeds[i, hs_start : hs_start + target_len, :]
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

    final_embeds = torch.stack(padded_embeds)
    final_masks = torch.stack(padded_masks)

    # Final truncation happens after encoding + slicing + batch shaping.
    if max_sequence_length is not None and final_embeds.shape[1] > max_sequence_length:
        final_embeds = final_embeds[:, :max_sequence_length, :]
        final_masks = final_masks[:, :max_sequence_length]

    return final_embeds, final_masks


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
