# -*- coding: utf-8 -*-
"""
多粒度球 + 动态自适应决策边界：
结合 reproduce_MOGB 的多粒度球与 BoundaryLoss 的边界学习，
将固定半径改为可学习的、自适应边界（可选椭球变换）。
"""
import torch
from torch import nn
import torch.nn.functional as F
import math
from .OODsampler import OODSampler


class AdaptiveBoundaryLoss(nn.Module):
    def __init__(self, args, gb_centroids, gb_radii, gb_labels, device=None):
        super(AdaptiveBoundaryLoss, self).__init__()
        self.num_labels = args.num_labels
        self.beta = getattr(args, 'beta', 0.1)
        self.feat_dim = args.feat_dim
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.triangle = getattr(args, 'triangle', False)
        self.args = args
        self.shape = getattr(args, 'shape', 'ball')
        self.ood = getattr(args, 'ood', True)

        # 多粒度球：质心与球标签固定（来自 GBNR），半径改为可学习
        self.register_buffer("centroids", gb_centroids.to(self.device))
        self.register_buffer("ball_labels", gb_labels.to(self.device).long().clamp(0, self.num_labels - 1))
        self.N_balls = self.centroids.size(0)

        # 每个球一个可学习边界 delta，初始化为聚类半径（或略放大）
        delta_init = gb_radii.to(self.device).float().clamp(min=1e-3)
        self.delta = nn.Parameter(delta_init.clone())

        # 椭球：按类共享 L/U/D（每类一个决策边界形状）
        n_tri = self.feat_dim * (self.feat_dim - 1) // 2
        self.L = nn.Parameter(torch.zeros((self.num_labels, n_tri), device=self.device))
        if not self.triangle:
            self.U = nn.Parameter(torch.zeros((self.num_labels, n_tri), device=self.device))
        self.D = nn.Parameter(torch.ones((self.num_labels, self.feat_dim), device=self.device))

        self.OODSampler = OODSampler(args)

    def get_rotate_matrix(self):
        """按类得到旋转/缩放矩阵 [num_labels, feat_dim, feat_dim]。"""
        rotate_matrix = torch.zeros((self.num_labels, self.feat_dim, self.feat_dim), device=self.device)
        indices = torch.tril_indices(self.feat_dim, self.feat_dim, offset=-1)
        dia_indices = range(self.feat_dim)
        for i in range(self.num_labels):
            if self.shape == "ball":
                rotate_matrix[i, dia_indices, dia_indices] = self.D[i, 0]
            elif self.shape == "regular_ellipsoid":
                rotate_matrix[i, dia_indices, dia_indices] = self.D[i, dia_indices]
            else:
                rotate_matrix[i, indices[0], indices[1]] = self.L[i]
                if not self.triangle:
                    rotate_matrix[i, indices[1], indices[0]] = self.U[i]
                else:
                    rotate_matrix[i, indices[1], indices[0]] = self.L[i]
                rotate_matrix[i, dia_indices, dia_indices] = self.D[i, dia_indices]
        return rotate_matrix

    def _assign_nearest_ball(self, pooled_output, labels):
        """
        为每个样本分配其类别内最近的球。
        pooled_output: [B, D], labels: [B]
        返回: ball_ids [B], 以及对应 delta [B], centroid [B, D], 类索引 [B] 用于旋转矩阵
        """
        B = pooled_output.size(0)
        # 到所有球的距离 [B, N_balls]
        dist_all = torch.cdist(pooled_output, self.centroids, p=2)
        # 只考虑与样本同类的球
        same_class = (self.ball_labels.unsqueeze(0) == labels.unsqueeze(1))  # [B, N_balls]
        dist_all = torch.where(same_class, dist_all, torch.full_like(dist_all, float('inf')))
        ball_ids = dist_all.argmin(dim=1)  # [B]，若某行全 inf 则取 0
        # 若某样本所在类没有球，则 same_class 全 False，argmin 会取 0，需要避免用错 delta
        d = self.delta[ball_ids]
        c = self.centroids[ball_ids]
        class_for_rotate = self.ball_labels[ball_ids]  # [B]
        return ball_ids, d, c, class_for_rotate

    def forward(self, pooled_output, labels, get_rotate_x=False):
        """
        pooled_output: [B, feat_dim], labels: [B] 类标签 (0..num_labels-1)。
        返回: pos_loss.mean(), neg_loss.mean(), pos_num, neg_num, total_loss
        """
        rotate_matrix = self.get_rotate_matrix()  # [num_labels, D, D]

        ball_ids, d, c, class_for_rotate = self._assign_nearest_ball(pooled_output, labels)
        x = pooled_output - c  # [B, D]
        # 按样本的类取旋转矩阵 [B, D, D]
        R = rotate_matrix[class_for_rotate.long()]  # [B, feat_dim, feat_dim]
        rotate_x = torch.bmm(R, x.unsqueeze(2)).squeeze(2)  # [B, D]
        if get_rotate_x:
            return rotate_x

        euc_dis = torch.norm(rotate_x, 2, 1)
        in_mask = (d > euc_dis)
        out_mask = ~in_mask
        pos_loss = (euc_dis - d) * out_mask.float() + torch.exp(euc_dis - d) * in_mask.float()
        pos_mask = (euc_dis > d).float()
        pos_num = pos_mask.sum()
        neg_mask = (euc_dis < d).float()
        neg_num = neg_mask.sum()

        if not self.ood:
            neg_loss = (d - euc_dis) * neg_mask
        else:
            ood = self.OODSampler(pooled_output, labels, pooled_output.size(0))
            if ood is None:
                neg_loss = torch.zeros(1, device=self.device)
            else:
                if isinstance(ood, (list, tuple)):
                    ood = torch.stack(ood, dim=0)
                neg_loss = torch.zeros(ood.size(0), device=self.device)
                for k in range(self.num_labels):
                    ball_mask = (self.ball_labels == k)
                    if not ball_mask.any():
                        continue
                    # 该类所有球的质心与 delta：对 OOD 取到该类最近球的距离
                    c_k = self.centroids[ball_mask]   # [n_k, D]
                    d_k = self.delta[ball_mask]      # [n_k]
                    dist_ood_to_balls = torch.cdist(ood, c_k, p=2)  # [N_ood, n_k]
                    nearest = dist_ood_to_balls.argmin(dim=1)       # [N_ood]
                    c_nearest = c_k[nearest]                         # [N_ood, D]
                    d_nearest = d_k[nearest]                        # [N_ood]
                    x_ood = ood - c_nearest
                    R_k = rotate_matrix[k].unsqueeze(0).expand(ood.size(0), -1, -1)
                    rotate_x_ood = torch.bmm(R_k, x_ood.unsqueeze(2)).squeeze(2)
                    euc_ood = torch.norm(rotate_x_ood, 2, 1)
                    in_ood = (d_nearest > euc_ood)
                    out_ood = ~in_ood
                    neg_loss += ((d_nearest - euc_ood + self.beta) * in_ood.float() +
                                 self.beta * torch.exp(d_nearest - euc_ood) * out_ood.float())

        loss = pos_loss.mean() + neg_loss.mean()
        return pos_loss.mean(), neg_loss.mean(), pos_num, neg_num, loss

    def get_delta_and_centroids(self):
        """返回当前学到的边界半径与质心（用于推理）。"""
        return self.delta.detach(), self.centroids, self.ball_labels
