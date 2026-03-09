from abc import ABC, abstractmethod


class BasePromptEnhancer(ABC):
    """
    Base class for all prompt enhancers.
    Subclasses must implement the `predict` method.
    Optionally override `predict_batch` for batch inference support.
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
        Enhance a single prompt.

        Returns:
            tuple: (reasoning_process, enhanced_prompt, marked_output)
        """
        raise NotImplementedError

    def predict_batch(
        self,
        prompts: list[str],
        sys_prompt: str = "",
        temperature: float = 0,
        top_p: float = 1.0,
        max_new_tokens: int = 512,
    ) -> list[tuple[str, str, str]]:
        """
        Enhance a batch of prompts. Default implementation calls predict() one by one.
        Subclasses (e.g. VLLMPromptEnhancer) can override for true batch inference.

        Returns:
            List of tuples: [(reasoning_process, enhanced_prompt, marked_output), ...]
        """
        return [
            self.predict(p, sys_prompt, temperature, top_p, max_new_tokens)
            for p in prompts
        ]