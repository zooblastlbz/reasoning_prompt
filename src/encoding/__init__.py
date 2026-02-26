import types
from .qwen_image_patch import encode_prompt_wrapper

def apply_monkey_patch(pipeline):
    """
    Applies the monkey patch to the given pipeline instance.
    Replaces the encode_prompt method with our custom wrapper.
    """
    pipeline.encode_prompt = types.MethodType(encode_prompt_wrapper, pipeline)
    return pipeline

__all__ = ["apply_monkey_patch"]