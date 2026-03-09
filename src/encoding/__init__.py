def get_encoder(model_type: str, method: str = "with_reasoning"):
    """
    Get the reasoning encoder function by model type and method.

    Args:
        model_type: Name of the diffusion model. Supported: "qwen_image".
        method: Encoding method. Supported: "with_reasoning", "weighted_reasoning".

    Returns:
        The encoder function for the given model type and method.
    """
    if model_type == "qwen_image":
        if method == "with_reasoning":
            from .qwen_image_patch import encode_with_reasoning
            return encode_with_reasoning
        elif method == "weighted_reasoning":
            from .qwen_image_patch import encode_with_weighted_reasoning
            return encode_with_weighted_reasoning
        else:
            raise ValueError(
                f"Unknown method: '{method}'. Supported: 'with_reasoning', 'weighted_reasoning'."
            )
    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. Supported: 'qwen_image'."
        )


__all__ = ["get_encoder"]