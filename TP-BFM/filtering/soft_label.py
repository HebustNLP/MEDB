
import torch
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

from tqdm import tqdm

logger = logging.getLogger(__name__)


class SoftLabelGenerator:
  

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "cuda",
        use_openai: bool = False,
        openai_model: str = "gpt-3.5-turbo",
        openai_api_key: Optional[str] = None,
    ):
        self.device = device
        self.use_openai = use_openai

        if use_openai:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=openai_api_key)
            self.openai_model = openai_model
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_soft_labels(
        self,
        texts: List[str],
        label_list: List[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(label_list)), dtype=np.float32)

        all_soft_labels = []

        for i in tqdm(range(0, len(texts), batch_size), desc="生成软标签"):
            batch_texts = texts[i: i + batch_size]
            batch_labels = self._compute_batch_soft_labels(batch_texts, label_list)
            all_soft_labels.append(batch_labels)

        if not all_soft_labels:
            return np.zeros((0, len(label_list)), dtype=np.float32)
        return np.concatenate(all_soft_labels, axis=0)

    def _compute_batch_soft_labels(
        self, texts: List[str], label_list: List[str]
    ) -> np.ndarray:
       
        if self.use_openai:
            return self._compute_openai_soft_labels(texts, label_list)
        else:
            return self._compute_local_soft_labels(texts, label_list)

    def _compute_local_soft_labels(
        self, texts: List[str], label_list: List[str]
    ) -> np.ndarray:
        # 构建标签-词元一一映射 V
        label_token_ids = self._build_label_verbalizer(label_list)

        soft_labels = np.zeros((len(texts), len(label_list)))

        for idx, text in enumerate(texts):
            # 构建任务提示 J(x, S)
            prompt = self._build_classification_prompt(text, label_list)

            model_device = next(self.model.parameters()).device
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model_device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # 获取最后一个 token 位置的 logits
                next_token_logits = outputs.logits[0, -1, :]

            # 提取标签词元对应的 logits 并归一化
            label_logits = next_token_logits[label_token_ids].float()
            label_probs = torch.softmax(label_logits, dim=0).cpu().numpy()
            soft_labels[idx] = label_probs

        return soft_labels

    def _build_label_verbalizer(self, label_list: List[str]) -> List[int]:
        """构建标签词具体化器 V: C -> vocab。"""
        token_ids: List[int] = []
        used = set()
        for label in label_list:
            # 对生成模型，前置空格通常更接近独立词元边界
            tokens = self.tokenizer.encode(" " + label, add_special_tokens=False)
            if not tokens:
                raise ValueError(f"标签无法映射到词元: {label}")
            tok = None
            for candidate in tokens:
                if candidate not in used:
                    tok = candidate
                    break
            if tok is None:
                # 回退尝试不带前置空格的 token 序列
                alt = self.tokenizer.encode(label, add_special_tokens=False)
                for candidate in alt:
                    if candidate not in used:
                        tok = candidate
                        break
            if tok is None:
                raise ValueError(f"标签词元冲突，无法构建一一映射: {label}")
            used.add(tok)
            token_ids.append(tok)
        return token_ids

    @staticmethod
    def _build_classification_prompt(text: str, label_list: List[str]) -> str:
        return (
            "Classify the following text into one of these categories: "
            f"{', '.join(label_list)}\n"
            f"Text: \"{text}\"\n"
            "Category:"
        )

    def _compute_openai_soft_labels(
        self, texts: List[str], label_list: List[str]
    ) -> np.ndarray:
      
        soft_labels = np.zeros((len(texts), len(label_list)))

        for idx, text in enumerate(texts):
            prompt = (
                f"Classify the following text into exactly one of these categories: "
                f"{', '.join(label_list)}\n"
                f"Text: \"{text}\"\n"
                f"Respond with only the category name."
            )

            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=20,
                    logprobs=True,
                    top_logprobs=20,
                )
                predicted = response.choices[0].message.content.strip()

                # 基于预测结果构建近似软标签
                for j, label in enumerate(label_list):
                    if label.lower() in predicted.lower():
                        soft_labels[idx, j] = 1.0
                        break
                else:
                    soft_labels[idx] = 1.0 / len(label_list)

            except Exception as e:
                logger.warning(f"OpenAI API 调用失败: {e}")
                soft_labels[idx] = 1.0 / len(label_list)

        # 归一化
        row_sums = soft_labels.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        soft_labels = soft_labels / row_sums

        return soft_labels