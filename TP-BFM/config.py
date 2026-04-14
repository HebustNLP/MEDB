"""
TP-BFM 项目配置文件
包含所有超参数、路径配置和实验设置
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """数据相关配置"""
    dataset_name: str = "oos"  # 数据集名称: oos
    data_dir: str = "./data/raw"
    output_dir: str = "./output"
    max_seq_length: int = 64
    known_class_ratio: float = 0.25  # 已知类别比例: 0.25, 0.50, 0.75
    num_seeds: int = 5  # 随机种子数量


@dataclass
class AugmentationConfig:
    """数据增强相关配置"""
    llm_model_name: str = "meta-llama/Llama-2-7b-chat-hf"  # 用于生成的大模型
    num_samples_per_class: int = 100  # 每个类别生成的样本数
    num_examples_per_prompt: int = 5  # 每个提示中包含的示例数 (k)
    top_p: float = 1.0
    temperature: float = 1.0
    frequency_penalty: float = 0.02
    max_new_tokens: int = 128
    use_openai_api: bool = False  # 是否使用 OpenAI API
    openai_model: str = "gpt-3.5-turbo"
    openai_api_key: Optional[str] = None


@dataclass
class FilterConfig:
    """过滤机制相关配置"""
    # 软标签置信度过滤
    confidence_threshold: float = 0.8  # δ: 置信度阈值
    label_mismatch_strategy: str = "drop"  # 不一致样本处理: drop / correct
    # PVI 过滤
    pvi_model_name: str = "bert-base-uncased"
    pvi_learning_rate: float = 2e-5
    pvi_epochs: int = 5
    pvi_batch_size: int = 32
    pvi_use_class_adaptive_threshold: bool = True  # 是否使用类别自适应阈值
    pvi_global_threshold: float = 0.0  # 全局 PVI 阈值 (当不使用自适应阈值时)


@dataclass
class ModelConfig:
    """模型相关配置"""
    backbone: str = "bert-base-uncased"
    hidden_dim: int = 768  # BERT 隐层维度 H
    feat_dim: int = 768  # 意图表征维度 D
    dropout: float = 0.1





@dataclass
class Config:
    """总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        os.makedirs(self.data.output_dir, exist_ok=True)
        os.makedirs(self.training.save_dir, exist_ok=True)