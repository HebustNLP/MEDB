
import json
import os
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class IntentDataset(Dataset):
    """意图检测数据集"""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 64,
        label_names: Optional[List[str]] = None,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_names = label_names

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_oos_dataset(data_dir: str) -> Dict:
   
    data_path = os.path.join(data_dir, "oos")

    splits = {}
    for split_name in ["train", "val", "test"]:
        file_path = os.path.join(data_path, f"{split_name}.tsv")
        texts, labels_str = [], []

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        texts.append(parts[0])
                        labels_str.append(parts[1])
        else:
            # 尝试加载 JSON 格式
            json_path = os.path.join(data_path, "data_full.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data.get(split_name, []):
                    texts.append(item[0])
                    labels_str.append(item[1])

        splits[split_name] = {"texts": texts, "labels_str": labels_str}

    # 构建标签映射 (不包含 oos 标签)
    all_labels = set()
    for split in splits.values():
        for label in split["labels_str"]:
            if label != "oos":
                all_labels.add(label)

    label_list = sorted(list(all_labels))
    label2id = {label: idx for idx, label in enumerate(label_list)}
    label2id["oos"] = -1  # OOS 标签设为 -1

    # 转换标签为 ID
    for split in splits.values():
        split["labels"] = [label2id.get(l, -1) for l in split["labels_str"]]

    return {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "label_list": label_list,
        "label2id": label2id,
        "id2label": {v: k for k, v in label2id.items()},
    }


def split_known_unknown(
    dataset: Dict,
    known_class_ratio: float,
    seed: int = 42,
) -> Dict:
 
    random.seed(seed)
    np.random.seed(seed)

    label_list = dataset["label_list"]
    num_known = int(len(label_list) * known_class_ratio)

    # 随机选择已知类别
    shuffled_labels = label_list.copy()
    random.shuffle(shuffled_labels)
    known_labels = set(shuffled_labels[:num_known])
    unknown_labels = set(shuffled_labels[num_known:])

    label2id = dataset["label2id"]

    # 已知类别重新编号
    known_label_list = sorted(list(known_labels))
    known_label2id = {label: idx for idx, label in enumerate(known_label_list)}
    open_class_id = len(known_label_list)  # 开放类 ID

    def filter_split(split_data, include_unknown=False):
        """过滤数据集"""
        texts, labels = [], []
        for text, label_str in zip(split_data["texts"], split_data["labels_str"]):
            if label_str in known_labels:
                texts.append(text)
                labels.append(known_label2id[label_str])
            elif include_unknown and label_str != "oos":
                texts.append(text)
                labels.append(open_class_id)
            elif include_unknown and label_str == "oos":
                texts.append(text)
                labels.append(open_class_id)
        return {"texts": texts, "labels": labels}

    # 训练集和验证集仅包含已知类别
    train_data = filter_split(dataset["train"], include_unknown=False)
    val_data = filter_split(dataset["val"], include_unknown=False)
    # 测试集包含已知和未知类别
    test_data = filter_split(dataset["test"], include_unknown=True)

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "known_label_list": known_label_list,
        "known_label2id": known_label2id,
        "num_known_classes": len(known_label_list),
        "open_class_id": open_class_id,
        "known_labels": known_labels,
        "unknown_labels": unknown_labels,
    }


def build_class_samples(texts: List[str], labels_str: List[str]) -> Dict[str, List[str]]:
  
    class_samples = defaultdict(list)
    for text, label in zip(texts, labels_str):
        if label != "oos":
            class_samples[label].append(text)
    return dict(class_samples)