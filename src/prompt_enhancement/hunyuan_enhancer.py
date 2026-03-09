import re
import logging
import torch
from .base_enhancer import BasePromptEnhancer


def replace_single_quotes(text):
    """
    Replace single quotes within words with double quotes, and convert
    curly single quotes to curly double quotes for consistency.
    """
    pattern = r"\B'([^']*)'\B"
    replaced_text = re.sub(pattern, r'"\1"', text)
    replaced_text = replaced_text.replace("\u2018", "\u201c")
    replaced_text = replaced_text.replace("\u2019", "\u201d")
    return replaced_text


def parse_output(output_res: str, org_prompt: str) -> tuple[str, str, str]:
    """
    Parse model output to extract reasoning, enhanced prompt, and marked output.

    Returns:
        tuple: (reasoning_process, enhanced_prompt, marked_output)
    """
    reasoning_process = ""

    # Extract reasoning process (inside <think> tags)
    think_pattern = r"<think>(.*?)</think>"
    think_matches = re.findall(think_pattern, output_res, re.DOTALL)
    if think_matches:
        reasoning_process = think_matches[0].strip()

    # Extract enhanced prompt (inside <answer> tags or fallback)
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_matches = re.findall(answer_pattern, output_res, re.DOTALL)
    if answer_matches:
        enhanced_prompt = answer_matches[0].strip()
    else:
        output_clean = re.sub(r"<think>[\s\S]*?</think>", "", output_res)
        output_clean = output_clean.strip()
        enhanced_prompt = output_clean if output_clean else org_prompt

    enhanced_prompt = replace_single_quotes(enhanced_prompt)

    # Mark the enhanced prompt in the full output
    marked_output = output_res.replace(
        enhanced_prompt,
        f"<enhanced_prompt>{enhanced_prompt}</enhanced_prompt>",
    )
    if "<enhanced_prompt>" not in marked_output:
        marked_output = f"{output_res}\n<enhanced_prompt>{enhanced_prompt}</enhanced_prompt>"

    return reasoning_process, enhanced_prompt, marked_output


class HunyuanPromptEnhancer(BasePromptEnhancer):

    def __init__(
        self,
        models_root_path: str,
        backend: str = "transformers",
        device_map: str = "auto",
        # vllm specific
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
    ):
        """
        Initialize the HunyuanPromptEnhancer.

        Args:
            models_root_path: Path to the pretrained model.
            backend: "transformers" for HuggingFace inference, "vllm" for vLLM offline batch inference.
            device_map: Device mapping (transformers only).
            tensor_parallel_size: Number of GPUs for tensor parallelism (vllm only).
            gpu_memory_utilization: GPU memory utilization ratio (vllm only).
            max_model_len: Maximum model context length (vllm only).
        """
        if not logging.getLogger(__name__).handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.backend = backend

        if backend == "transformers":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model = AutoModelForCausalLM.from_pretrained(
                models_root_path, device_map=device_map, trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                models_root_path, trust_remote_code=True
            )
        elif backend == "vllm":
            from vllm import LLM

            self.llm = LLM(
                model=models_root_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                trust_remote_code=True,
            )
            self.tokenizer = self.llm.get_tokenizer()
        else:
            raise ValueError(f"Unknown backend: '{backend}'. Supported: 'transformers', 'vllm'.")

    def _build_chat_prompt(self, prompt: str, sys_prompt: str) -> str:
        """Build the chat prompt string using the tokenizer's chat template."""
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # ---- transformers backend ----

    @torch.inference_mode()
    def _predict_transformers(
        self, prompt, sys_prompt, temperature, top_p, max_new_tokens,
    ) -> tuple[str, str, str]:
        org_prompt = prompt
        try:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": org_prompt},
            ]
            tokenized_chat = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )
            if isinstance(tokenized_chat, torch.Tensor):
                inputs = tokenized_chat.to(self.model.device)
            else:
                inputs = tokenized_chat["input_ids"].to(self.model.device)

            do_sample = temperature is not None and float(temperature) > 0
            outputs = self.model.generate(
                inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=do_sample,
                temperature=float(temperature) if do_sample else None,
                top_p=float(top_p) if do_sample else None,
            )

            generated_sequence = outputs[0]
            prompt_length = inputs.shape[-1]
            new_tokens = generated_sequence[prompt_length:]
            output_res = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            reasoning, enhanced, marked = parse_output(output_res, org_prompt)
            self.logger.info("Re-prompting succeeded; using the new prompt")
            return reasoning, enhanced, marked
        except Exception:
            self.logger.exception("Re-prompting failed; using the original prompt")
            return "", org_prompt, f"<enhanced_prompt>{org_prompt}</enhanced_prompt>"

    # ---- vllm backend ----

    def _predict_batch_vllm(
        self, prompts, sys_prompt, temperature, top_p, max_new_tokens,
    ) -> list[tuple[str, str, str]]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=float(temperature) if temperature > 0 else 0,
            top_p=float(top_p) if temperature > 0 else 1.0,
            max_tokens=int(max_new_tokens),
        )

        chat_prompts = [self._build_chat_prompt(p, sys_prompt) for p in prompts]
        outputs = self.llm.generate(chat_prompts, sampling_params)

        results = []
        for i, output in enumerate(outputs):
            output_res = output.outputs[0].text
            try:
                reasoning, enhanced, marked = parse_output(output_res, prompts[i])
            except Exception:
                self.logger.exception(f"Failed to parse output for prompt {i}. Using original.")
                reasoning = ""
                enhanced = prompts[i]
                marked = f"<enhanced_prompt>{prompts[i]}</enhanced_prompt>"
            results.append((reasoning, enhanced, marked))

        return results

    # ---- public interface ----

    def predict(
        self,
        prompt,
        sys_prompt="你是一位图像生成提示词撰写专家，请根据用户输入的提示词，改写生成新的提示词，改写后的提示词要求：1 改写后提示词包含的主体/动作/数量/风格/布局/关系/属性/文字等 必须和改写前的意图一致； 2 在宏观上遵循\u201c总-分-总\u201d的结构，确保信息的层次清晰；3 客观中立，避免主观臆断和情感评价；4 由主到次，始终先描述最重要的元素，再描述次要和背景元素；5 逻辑清晰，严格遵循空间逻辑或主次逻辑，使读者能在大脑中重建画面；6 结尾点题，必须用一句话总结图像的整体风格或类型。",
        temperature=0,
        top_p=1.0,
        max_new_tokens=512,
    ) -> tuple[str, str, str]:
        if self.backend == "transformers":
            return self._predict_transformers(prompt, sys_prompt, temperature, top_p, max_new_tokens)
        else:
            return self._predict_batch_vllm([prompt], sys_prompt, temperature, top_p, max_new_tokens)[0]

    def predict_batch(
        self,
        prompts: list[str],
        sys_prompt="你是一位图像生成提示词撰写专家，请根据用户输入的提示词，改写生成新的提示词，改写后的提示词要求：1 改写后提示词包含的主体/动作/数量/风格/布局/关系/属性/文字等 必须和改写前的意图一致； 2 在宏观上遵循\u201c总-分-总\u201d的结构，确保信息的层次清晰；3 客观中立，避免主观臆断和情感评价；4 由主到次，始终先描述最重要的元素，再描述次要和背景元素；5 逻辑清晰，严格遵循空间逻辑或主次逻辑，使读者能在大脑中重建画面；6 结尾点题，必须用一句话总结图像的整体风格或类型。",
        temperature=0,
        top_p=1.0,
        max_new_tokens=512,
    ) -> list[tuple[str, str, str]]:
        if self.backend == "vllm":
            return self._predict_batch_vllm(prompts, sys_prompt, temperature, top_p, max_new_tokens)
        else:
            # transformers fallback: sequential
            return [
                self._predict_transformers(p, sys_prompt, temperature, top_p, max_new_tokens)
                for p in prompts
            ]