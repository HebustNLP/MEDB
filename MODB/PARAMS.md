# MOGB 方法参数说明

本文档说明 **多粒度粒球（MOGB）** 中全部可配置参数，重点说明**纯度**与**球分裂**如何通过参数控制。

---

## 一、纯度与球分裂（粒球聚类过程）

粒球由“纯度约束 + 最小球规模”下的迭代分裂得到，所有相关阈值均为可调参数。

### 1. 分裂阶段参数（何时继续分裂）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `purity_train` | float | 0.9 | **训练阶段**粒球纯度阈值。当某球内**主类占比 < purity_train** 且 **样本数 > min_ball_train** 时，对该球执行一次分裂。 |
| `min_ball_train` | int | 3 | **训练阶段**允许分裂的最小球规模。样本数 ≤ 此值的球不再因低纯度而分裂。 |
| `purity_get_ball` | float | 0.9 | **计算最终粒球阶段**的纯度阈值，规则同 `purity_train`，一般可设更严格（如 0.95）。 |
| `min_ball_get_ball` | int | 3 | **计算最终粒球阶段**的最小球规模，规则同 `min_ball_train`。 |

**纯度定义**：对某个粒球，纯度 = 该球内“主类”（出现次数最多的类别）样本数 / 球内总样本数，取值 (0, 1]。纯度越高表示球内类别越单一。

**分裂逻辑**：  
`if 当前球纯度 < 对应 purity 且 当前球样本数 > 对应 min_ball: 调用 splits_ball() 分裂该球`，直到没有球再满足条件。

### 2. 选球阶段参数（分裂完成后保留哪些球）

分裂结束后，只保留“纯度足够高”且“样本数足够多”的球用于质心/半径与后续分类。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `purity_select_ball_train` | float | 0.1 | **训练阶段**保留粒球的纯度下界。纯度 **>** 此值的球才参与训练时的最近子质心损失（默认 0.1 相当于几乎全保留）。 |
| `min_ball_select_ball_train` | int | 1 | **训练阶段**保留粒球的最小样本数，样本数 **≥** 此值才保留。 |
| `purity_select_ball` | float | 0.8 | **最终粒球**保留的纯度下界。纯度 **>** 此值的球才用于验证/测试与开放集推理。 |
| `min_ball_select_ball` | int | 2 | **最终粒球**保留的最小样本数。 |

因此：**球分裂过程**由 `purity_train/get_ball` 与 `min_ball_train/get_ball` 控制；**保留哪些球**由 `purity_select_ball_train/select_ball` 与 `min_ball_select_ball_train/select_ball` 控制。

---

## 二、数据与任务

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_dir` | str | "data" | 数据根目录。 |
| `dataset` | str | "oos" | 数据集名称（如 oos, clinc, stackoverflow 等）。 |
| `known_cls_ratio` | float | 0.75 | 已知类占比 (0~1)。 |
| `labeled_ratio` | float | 1.0 | 训练时使用的标注比例 (0~1)。 |
| `seed` | int | 42 | 随机种子。 |

---

## 三、模型与训练

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `bert_model` | str | "bert-base-uncased" | BERT 预训练模型名或路径。 |
| `freeze_bert_parameters` | bool | False | 是否冻结 BERT 大部分参数（仅解冻顶层等）。 |
| `feat_dim` | int | 768 | 特征维度，需与 BERT 输出一致。 |
| `gpu_id` | str | "0" | 使用的 GPU 编号。 |
| `train_batch_size` | int | 32 | 训练 batch 大小。 |
| `eval_batch_size` | int | 64 | 验证/测试 batch 大小。 |
| `num_train_epochs` | int | 10 | 训练轮数。 |
| `wait_patient` | int | 5 | 验证集指标无提升的容忍轮数，超过则早停。 |
| `save_model` | bool | False | 是否保存预训练权重。 |
| `pretrain_dir` | str | "saved" | 预训练模型保存目录。 |
| `save_results_path` | str | "results" | 测试结果与混淆矩阵保存目录。 |

---

## 四、优化器与学习率

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lr` | float | 2e-5 | BERT 编码器学习率。 |
| `lr2` | float | 1e-4 | 粒球/分类头相关参数的学习率。 |
| `warmup_proportion` | float | 0.1 | 学习率 warmup 步数占总步数比例。 |

---

## 五、动态自适应边界（可选）

在得到多粒度粒球后，可再训练“可学习边界”（替代固定半径）用于开放集推理。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lr3` | float | 1e-4 | 自适应边界模块学习率。 |
| `adaptive_boundary_epochs` | int | 0 | 若 >0，在粒球基础上训练可学习边界；0 表示仅用固定半径。 |
| `beta` | float | 0.1 | 边界损失中 OOD 边际项权重。 |
| `triangle` | bool | False | 椭球是否使用三角参数化（节省参数）。 |
| `shape` | str | "ball" | 决策边界形状：`"ball"` 为球，`"ellipse"` 为椭球。 |
| `ood` | bool | True | 边界损失是否包含 OOD 边际。 |

---

## 六、参数汇总表（纯度与球分裂）

| 阶段 | 纯度参数 | 最小球规模参数 | 作用 |
|------|----------|----------------|------|
| 训练时分裂 | `purity_train` | `min_ball_train` | 满足“纯度 < 阈值且样本数 > 规模”则分裂 |
| 最终粒球分裂 | `purity_get_ball` | `min_ball_get_ball` | 同上，用于得到最终粒球结构 |
| 训练时选球 | `purity_select_ball_train` | `min_ball_select_ball_train` | 只保留纯度 > 阈值且样本数 ≥ 规模的球 |
| 最终选球 | `purity_select_ball` | `min_ball_select_ball` | 只保留满足条件的球用于推理 |

通过调整上述 8 个参数，即可完整控制“纯度约束下的粒球分裂”与“选球”行为。
