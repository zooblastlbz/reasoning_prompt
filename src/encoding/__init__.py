def get_encoder(model_type: str):
    """
    Get the reasoning encoder function by model type.

    Args:
        model_type: Name of the diffusion model. Supported: "qwen_image".

    Returns:
        The encode_with_reasoning function for the given model type.
    """
    if model_type == "qwen_image":
        from .qwen_image_patch import encode_with_reasoning
        return encode_with_reasoning
    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. Supported: 'qwen_image'."
        )


__all__ = ["get_encoder"]