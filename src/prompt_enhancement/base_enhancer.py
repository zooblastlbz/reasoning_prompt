from abc import ABC, abstractmethod


class BasePromptEnhancer(ABC):
    """
    Base class for all prompt enhancers.
    Subclasses must implement the `predict` method.
    """

    @abstractmethod
    def predict(
        self,
        prompt: str,
        sys_prompt: str = "",
        temperature: float = 0,
        top_p: float = 1.0,
        max_new_tokens: int = 512,
    ) -> tuple[str, str, str]:
        """
        Enhance a prompt using the model.

        Args:
            prompt: The original prompt to be rewritten.
            sys_prompt: System prompt to guide the rewriting.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            tuple: (reasoning_process, enhanced_prompt, marked_output)
                - reasoning_process: The extracted reasoning/thinking process.
                - enhanced_prompt: The final enhanced prompt.
                - marked_output: Full model output with enhanced prompt marked by
                  <enhanced_prompt>...</enhanced_prompt> tags.
        """
        raise NotImplementedError