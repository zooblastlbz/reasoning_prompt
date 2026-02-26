from .base_enhancer import BasePromptEnhancer
from .templates import DEFAULT_SYS_PROMPT


def get_enhancer(name: str, **kwargs) -> BasePromptEnhancer:
    """
    Factory function to get a prompt enhancer by name.

    Args:
        name: Name of the enhancer. Supported: "hunyuan".
        **kwargs: Arguments passed to the enhancer constructor (e.g., models_root_path).

    Returns:
        An instance of the requested prompt enhancer.
    """
    if name == "hunyuan":
        from .hunyuan_enhancer import HunyuanPromptEnhancer
        return HunyuanPromptEnhancer(**kwargs)
    else:
        raise ValueError(
            f"Unknown enhancer: '{name}'. Supported: 'hunyuan'."
        )


__all__ = ["BasePromptEnhancer", "get_enhancer", "DEFAULT_SYS_PROMPT"]