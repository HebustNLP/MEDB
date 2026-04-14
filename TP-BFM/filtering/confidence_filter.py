"""
基于软标签的置信度过滤模块 (3.4.2)
通过置信度阈值剔除意图指向模糊或偏离目标类别的样本
"""

import numpy as np
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class ConfidenceFilter:
    """
    基于软标签置信度的过滤器

    论文公式 (3-7):
        conf(x') = max_{c∈C} q(c | x')

    仅保留满足 conf(x') > δ 的样本
    """

    def __init__(self, threshold: float = 0.5, label_mismatch_strategy: str = "drop"):
        """
        Args:
            threshold: 置信度阈值 δ
            label_mismatch_strategy:
                - "drop": 剔除生成标签与软标签预测不一致样本
                - "correct": 使用软标签预测标签进行修正
        """
        self.threshold = threshold
        if label_mismatch_strategy not in {"drop", "correct"}:
            raise ValueError("label_mismatch_strategy must be 'drop' or 'correct'")
        self.label_mismatch_strategy = label_mismatch_strategy

    def filter(
        self,
        texts: List[str],
        labels: List[str],
        soft_labels: np.ndarray,
        label_list: List[str],
    ) -> Tuple[List[str], List[str], np.ndarray]:
        """
        执行置信度过滤

        核心思想：如果软标签分布高度集中于某一类别，则该样本具有较高的标签可靠性；
        反之，如果分布过于分散，则表明意图指向模糊。

        Args:
            texts: 合成文本列表
            labels: 对应的生成标签列表
            soft_labels: 软标签概率矩阵 (N, num_classes)
            label_list: 所有类别名称列表

        Returns:
            过滤后的 (文本列表, 标签列表, 软标签矩阵)
        """
        assert len(texts) == len(labels) == soft_labels.shape[0]

        # 计算每个样本的最大类别置信度
        confidences = np.max(soft_labels, axis=1)

        # 检查生成标签与软标签预测是否一致
        predicted_labels = np.argmax(soft_labels, axis=1)

        filtered_texts = []
        filtered_labels = []
        filtered_soft_labels = []

        total = len(texts)
        kept = 0
        label_mismatch = 0

        for i in range(total):
            conf = confidences[i]
            pred_label_name = label_list[predicted_labels[i]]

            # 置信度过滤
            if conf <= self.threshold:
                continue

            # 检查生成标签与软标签预测一致性
            if pred_label_name != labels[i]:
                label_mismatch += 1
                if self.label_mismatch_strategy == "drop":
                    continue
                corrected_label = pred_label_name
            else:
                corrected_label = labels[i]

            filtered_texts.append(texts[i])
            filtered_labels.append(corrected_label)
            filtered_soft_labels.append(soft_labels[i])
            kept += 1

        logger.info(
            f"置信度过滤: 总计 {total} 样本, "
            f"保留 {kept} ({(kept/total*100) if total else 0.0:.1f}%), "
            f"标签不一致 {label_mismatch} 个"
        )

        if filtered_soft_labels:
            return (
                filtered_texts,
                filtered_labels,
                np.stack(filtered_soft_labels),
            )
        else:
            return [], [], np.array([])