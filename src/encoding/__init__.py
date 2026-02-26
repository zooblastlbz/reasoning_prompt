import types


def apply_patch(pipeline, model_type: str):
    """
    Apply the encode_prompt monkey patch to the given pipeline.

    Args:
        pipeline: The diffusion pipeline instance.
        model_type: Name of the diffusion model. Supported: "qwen_image".

    Returns:
        The patched pipeline instance.
    """
    if model_type == "qwen_image":
        from .qwen_image_patch import encode_prompt_wrapper
    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. Supported: 'qwen_image'."
        )

    pipeline.encode_prompt = types.MethodType(encode_prompt_wrapper, pipeline)
    return pipeline


__all__ = ["apply_patch"]