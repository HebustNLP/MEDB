

import os
import re
import logging
from typing import List, Dict, Tuple, Optional

import torch
from tqdm import tqdm

from .tuple_prompt import TuplePromptBuilder

logger = logging.getLogger(__name__)


class SampleGenerator:
   

    def __init__(
        self,
        prompt_builder: TuplePromptBuilder,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        use_openai: bool = False,
        openai_model: str = "gpt-3.5-turbo",
        openai_api_key: Optional[str] = None,
        device: str = "cuda",
        top_p: float = 1.0,
        temperature: float = 1.0,
        frequency_penalty: float = 0.02,
        max_new_tokens: int = 128,
    ):
        self.prompt_builder = prompt_builder
        self.use_openai = use_openai
        self.device = device
        self.top_p = top_p
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.max_new_tokens = max_new_tokens

        if use_openai:
            self._init_openai(openai_model, openai_api_key)
        else:
            self._init_local_model(model_name)

    def _init_openai(self, model: str, api_key: str):

        try:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.openai_model = model
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

    def _init_local_model(self, model_name: str):
        """初始化本地 HuggingFace 模型"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"加载模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_samples(
        self,
        class_name: str,
        seed_examples: List[str],
        num_samples: int = 100,
        batch_size: int = 10,
    ) -> List[str]:
   
        all_samples = []
        remaining = num_samples

        while remaining > 0:
            current_batch = min(batch_size, remaining)
            prompt = self.prompt_builder.build_prompt(
                class_name=class_name,
                examples=seed_examples,
                num_to_generate=current_batch,
            )

            if self.use_openai:
                generated_text = self._generate_openai(prompt)
            else:
                generated_text = self._generate_local(prompt)

            # 解析生成的文本为独立样本
            samples = self._parse_generated_text(generated_text)
            all_samples.extend(samples)
            remaining -= len(samples)

        return all_samples[:num_samples]

    def generate_for_all_classes(
        self,
        class_seeds: Dict[str, List[str]],
        num_samples_per_class: int = 100,
    ) -> Dict[str, List[str]]:
        
        all_generated = {}
        for class_name, seeds in tqdm(class_seeds.items(), desc="生成增强数据"):
            logger.info(f"为类别 '{class_name}' 生成 {num_samples_per_class} 个样本")
            samples = self.generate_samples(
                class_name=class_name,
                seed_examples=seeds,
                num_samples=num_samples_per_class,
            )
            all_generated[class_name] = samples
            logger.info(f"  实际生成 {len(samples)} 个样本")

        return all_generated

    def _generate_openai(self, prompt: str) -> str:
       
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            max_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content

    def _generate_local(self, prompt: str) -> str:
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=1.0 + self.frequency_penalty,
                do_sample=True,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    @staticmethod
    def _parse_generated_text(text: str) -> List[str]:
       
        lines = text.strip().split("\n")
        samples = []
        for line in lines:
            # 去除序号、特殊符号等
            cleaned = re.sub(r"^\d+[\.\)\]\-]\s*", "", line.strip())
            cleaned = cleaned.strip("- ").strip()
            if cleaned and len(cleaned) > 3:
                samples.append(cleaned)
        return samples