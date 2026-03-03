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

若出现 `numpy`/`accelerate` 相关报错，可执行：`pip install "numpy<2"` 或使用新的虚拟环境。

## 数据格式

每个数据集目录下需包含 `train.tsv`、`dev.tsv`、`test.tsv`，格式为两列（文本、标签），制表符分隔，首行为表头。

## 运行方式

### 1. 使用脚本（推荐）

在项目根目录下执行：

```bash
# StackOverflow 50% 已知类
bash scripts/run.sh

# StackOverflow 75% 已知类
bash scripts/runstackoverflow75.sh

# Banking 数据集
bash scripts/runbanking.sh

# OOS 数据集
bash scripts/runoos.sh

# 学习率调参
bash scripts/tune_lr.sh
```

### 2. 直接运行

```bash
python main.py \
  --dataset stackoverflow \
  --known_cls_ratio 0.5 \
  --labeled_ratio 1.0 \
  --seed 0 \
  --freeze_bert_parameters \
  --bert_model bert-base-uncased \
  --gpu_id 0 \
  --train_batch_size 128 \
  --eval_batch_size 128 \
  --wait_patient 10 \
  --pretrain_dir outputs/saved \
  --save_results_path outputs/results
```

### 3. BERT 模型路径

默认使用 HuggingFace 模型名 `bert-base-uncased`，会自动下载。若使用本地 BERT 权重：

```bash
export BERT_MODEL=/path/to/uncased_L-12_H-768_A-12
bash scripts/run.sh
```

或在命令行中传入 `--bert_model /path/to/bert`。

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | data | 数据根目录 |
| `--dataset` | oos | 数据集：oos, clinc, stackoverflow, banking, snips 等 |
| `--known_cls_ratio` | 0.75 | 已知类占比 (0~1) |
| `--bert_model` | bert-base-uncased | BERT 模型名或路径 |
| `--pretrain_dir` | outputs/saved | 模型保存目录 |
| `--save_results_path` | outputs/results | 结果保存目录 |
| `--adaptive_boundary_epochs` | 0 | 边界训练轮数，>0 时启用自适应边界 |

更多参数见 `init_parameter.py` 或 `PARAMS.md`。

## 输出说明

- **模型**：保存于 `outputs/saved/`（或 `--pretrain_dir` 指定路径）
- **结果**：`outputs/results/results.csv` 记录各次实验指标
- **混淆矩阵**：`outputs/results/{dataset}_{known_cls_ratio}_{seed}_{min_ball_train}.txt`

## 引用

若使用本代码，请引用对应论文。
