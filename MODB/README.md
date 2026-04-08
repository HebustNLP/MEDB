# MODB: Multi-Granularity Open Intent Classification via Adaptive Granular-Ball Decision Boundary

基于多粒度粒球与自适应决策边界的开放集意图分类方法。

## 项目结构

```
MODB/
├── main.py                 # 主入口
├── init_parameter.py       # 参数定义
├── src/                    # 核心代码
│   ├── model.py            # BERT 模型
│   ├── dataloader.py       # 数据加载
│   ├── pretrain.py         # 预训练与粒球计算
│   ├── gb_test.py          # 评估与开放集分类
│   ├── myloss.py           # 粒球聚类损失
│   ├── adaptive_boundary_loss.py  # 自适应边界损失
│   ├── OODsampler.py       # OOD 负样本采样
│   ├── util.py             # 工具函数
│   └── cluster/            # 粒球聚类
│       ├── cluster.py      # 粒球封装
│       ├── cluster2.py     # GBNR 粒球分裂
│       └── cluster3.py     # 粒球分裂核心逻辑
├── scripts/                # 运行脚本
│   ├── run.sh              # StackOverflow 50%
│   ├── runstackoverflow75.sh  # StackOverflow 75%
│   ├── runbanking.sh       # Banking 数据集
│   ├── runoos.sh           # OOS 数据集
│   └── tune_lr.sh          # 学习率调参
├── data/                   # 数据目录
│   ├── stackoverflow/
│   ├── banking/
│   ├── oos/
│   └── ...
├── outputs/                # 输出目录（模型与结果）
│   ├── saved/              # 预训练模型
│   └── results/            # 评估结果
├── docs/                   # 文档
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.8+
- PyTorch
- transformers
- 其他依赖见 `requirements.txt`

```bash
pip install -r requirements.txt
```

