

import copy
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SimpleTextDataset(Dataset):
    """简单文本分类数据集"""

    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class PVIFilter:
  

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 150,
        learning_rate: float = 2e-5,
        epochs: int = 5,
        batch_size: int = 32,
        max_length: int = 64,
        device: str = "cuda",
        use_class_adaptive_threshold: bool = True,
        global_threshold: float = 0.0,
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.use_class_adaptive_threshold = use_class_adaptive_threshold
        self.global_threshold = global_threshold

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.null_token = "[PAD]"  # 空白占位符

    def train_models(
        self,
        train_texts: List[str],
        train_labels: List[int],
    ) -> Tuple[nn.Module, nn.Module]:
       
        logger.info("训练有输入模型 g'...")
        model_with_input = self._train_single_model(
            texts=train_texts,
            labels=train_labels,
            use_null_input=False,
        )

        logger.info("训练空输入模型 g...")
        model_null_input = self._train_single_model(
            texts=train_texts,
            labels=train_labels,
            use_null_input=True,
        )

        return model_with_input, model_null_input

    def _train_single_model(
        self,
        texts: List[str],
        labels: List[int],
        use_null_input: bool = False,
    ) -> nn.Module:
        """训练单个分类模型"""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_classes
        ).to(self.device)

        # 空输入模型：将所有输入替换为空白占位符
        if use_null_input:
            texts = [self.null_token] * len(texts)

        dataset = SimpleTextDataset(texts, labels, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                label = batch["label"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        model.eval()
        return model

    def compute_pvi(
        self,
        texts: List[str],
        labels: List[int],
        model_with_input: nn.Module,
        model_null_input: nn.Module,
    ) -> np.ndarray:
       
        pvi_values = []

        dataset = SimpleTextDataset(texts, labels, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        null_dataset = SimpleTextDataset(
            [self.null_token] * len(texts), labels, self.tokenizer, self.max_length
        )
        null_dataloader = DataLoader(null_dataset, batch_size=self.batch_size, shuffle=False)

        model_with_input.eval()
        model_null_input.eval()

        with torch.no_grad():
            for batch, null_batch in zip(dataloader, null_dataloader):
                # 有输入模型预测
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["label"].to(self.device)

                outputs_with = model_with_input(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                probs_with = torch.softmax(outputs_with.logits, dim=-1)

                # 空输入模型预测
                null_input_ids = null_batch["input_ids"].to(self.device)
                null_attention_mask = null_batch["attention_mask"].to(self.device)

                outputs_null = model_null_input(
                    input_ids=null_input_ids, attention_mask=null_attention_mask
                )
                probs_null = torch.softmax(outputs_null.logits, dim=-1)

                # 计算 PVI
                for i in range(len(batch_labels)):
                    c = batch_labels[i].item()
                    g_c = probs_null[i, c].item()
                    g_prime_c = probs_with[i, c].item()

                    # 防止 log(0)
                    g_c = max(g_c, 1e-10)
                    g_prime_c = max(g_prime_c, 1e-10)

                    pvi = -np.log2(g_c) + np.log2(g_prime_c)
                    pvi_values.append(pvi)

        return np.array(pvi_values)

    def compute_class_thresholds(
        self,
        pvi_values: np.ndarray,
        labels: List[int],
        percentile: float = 25.0,
    ) -> Dict[int, float]:
      
        class_pvi = defaultdict(list)
        for pvi, label in zip(pvi_values, labels):
            class_pvi[label].append(pvi)

        thresholds = {}
        for class_id, values in class_pvi.items():
            thresholds[class_id] = np.percentile(values, percentile)

        return thresholds

    def filter(
        self,
        texts: List[str],
        labels: List[int],
        label_names: List[str],
        train_texts: List[str],
        train_labels: List[int],
    ) -> Tuple[List[str], List[int]]:
      
        # 训练两个对比模型
        model_with_input, model_null_input = self.train_models(
            train_texts, train_labels
        )

        # 计算 PVI 值
        logger.info("计算合成样本的 PVI 值...")
        pvi_values = self.compute_pvi(
            texts, labels, model_with_input, model_null_input
        )

        # 确定阈值
        if self.use_class_adaptive_threshold:
            thresholds = self.compute_class_thresholds(pvi_values, labels)
            logger.info(f"类别自适应阈值: {thresholds}")
        else:
            thresholds = {c: self.global_threshold for c in set(labels)}

        # 过滤
        filtered_texts = []
        filtered_labels = []
        total = len(texts)

        for i in range(total):
            class_id = labels[i]
            threshold = thresholds.get(class_id, self.global_threshold)
            if pvi_values[i] > threshold:
                filtered_texts.append(texts[i])
                filtered_labels.append(labels[i])

        kept = len(filtered_texts)
        logger.info(
            f"PVI 过滤: 总计 {total} 样本, 保留 {kept} ({kept/total*100:.1f}%)"
        )

        return filtered_texts, filtered_labels