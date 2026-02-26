import re
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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


class HunyuanPromptEnhancer(BasePromptEnhancer):

    def __init__(self, models_root_path, device_map="auto"):
        """
        Initialize the HunyuanPromptEnhancer class with model and processor.

        Args:
            models_root_path (str): Path to the pretrained model.
            device_map (str): Device mapping for model loading.
        """
        if not logging.getLogger(__name__).handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.model = AutoModelForCausalLM.from_pretrained(
            models_root_path, device_map=device_map, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            models_root_path, trust_remote_code=True
        )

    @torch.inference_mode()
    def predict(
        self,
        prompt,
        sys_prompt="你是一位图像生成提示词撰写专家，请根据用户输入的提示词，改写生成新的提示词，改写后的提示词要求：1 改写后提示词包含的主体/动作/数量/风格/布局/关系/属性/文字等 必须和改写前的意图一致； 2 在宏观上遵循\u201c总-分-总\u201d的结构，确保信息的层次清晰；3 客观中立，避免主观臆断和情感评价；4 由主到次，始终先描述最重要的元素，再描述次要和背景元素；5 逻辑清晰，严格遵循空间逻辑或主次逻辑，使读者能在大脑中重建画面；6 结尾点题，必须用一句话总结图像的整体风格或类型。",
        temperature=0,
        top_p=1.0,
        max_new_tokens=512,
    ):
        """
        Generate a rewritten prompt using the model.

        Returns:
            tuple: (reasoning_process, enhanced_prompt, marked_output)
        """
        org_prompt = prompt
        reasoning_process = ""
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
            inputs = tokenized_chat.to(self.model.device)
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

            self.logger.info("Re-prompting succeeded; using the new prompt")
        except Exception as e:
            enhanced_prompt = org_prompt
            marked_output = f"<enhanced_prompt>{org_prompt}</enhanced_prompt>"
            self.logger.exception("Re-prompting failed; using the original prompt")

        return reasoning_process, enhanced_prompt, marked_output