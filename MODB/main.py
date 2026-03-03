#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MODB 主入口：多粒度粒球 + 开放集意图分类
"""
import random
import numpy as np
import torch

from init_parameter import init_model
from src.dataloader import Data
from src.pretrain import PretrainModelManager
from src.gb_test import ModelManager


if __name__ == '__main__':
    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    print('Parameters Initialization...')

    parser = init_model()
    args = parser.parse_args()
    data = Data(args)

    if torch.cuda.is_available():
        print(f'当前使用: GPU (cuda:{args.gpu_id}, device={torch.cuda.get_device_name(0)})')
    else:
        print('当前使用: CPU（未检测到可用 GPU，run.sh 中的 gpu_id 仅在存在 GPU 时生效）')

    print('Training begin...')
    manager_p1 = PretrainModelManager(args, data)
    manager_p1.train(args, data)
    print('Training finished!')

    print('Calculate ball begin...')
    gb_centroids, gb_radii, gb_labels = manager_p1.calculate_granular_balls(args, data)
    print('Calculate ball finished!')

    boundary_loss = None
    if getattr(args, 'adaptive_boundary_epochs', 0) > 0:
        print('Train adaptive boundary begin...')
        gb_centroids, gb_radii, gb_labels, boundary_loss = manager_p1.train_adaptive_boundary(
            args, data, gb_centroids, gb_radii, gb_labels)
        print('Train adaptive boundary finished!')

    manager = ModelManager(args, data, manager_p1.model)
    print('Evaluation begin...')
    manager.evaluation(args, data, gb_centroids, gb_radii, gb_labels, mode="test", boundary_loss=boundary_loss)
    print('Evaluation finished!')
