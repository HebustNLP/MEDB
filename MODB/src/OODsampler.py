# -*- coding: utf-8 -*-
"""简单 OOD 采样器：从当前 batch 中采其他类样本作为负样本（边界外）"""
import torch


class OODSampler:
    def __init__(self, args):
        self.num_labels = getattr(args, 'num_labels', 10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, pooled_output, labels, batch_size):
        """
        从 batch 中采"其他类"样本作为 OOD 负样本。
        pooled_output: [B, D], labels: [B]
        返回: [N_ood, D] 或 None（若当前 batch 仅有一类）
        """
        if pooled_output is None or pooled_output.size(0) == 0:
            return None
        device = pooled_output.device
        ood_list = []
        for i in range(pooled_output.size(0)):
            other_class_mask = (labels != labels[i])
            if other_class_mask.any():
                idx = torch.where(other_class_mask)[0]
                j = idx[torch.randint(len(idx), (1,), device=device)].item()
                ood_list.append(pooled_output[j])
        if len(ood_list) == 0:
            return None
        return torch.stack(ood_list, dim=0)
