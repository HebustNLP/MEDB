# ood 环境版本冲突修复说明

在 `ood` conda 环境中执行以下步骤可解决版本冲突并正常运行 MOGB。

## 1. 已执行过的修复（若你重装环境可参考）

### 1.1 scipy 降级（解决 _spropack 导入错误）
```bash
conda activate ood
pip install "scipy>=1.11,<1.15" "numpy>=1.24,<2"
```
- 原问题：`scipy 1.17.0` 导致 `ImportError: cannot import name '_spropack'`
- 建议：将 scipy 固定在 1.11–1.14 之间（如 1.14.1），numpy 保持 1.x（如 1.26.4）

### 1.2 scikit-learn 重装（仅当出现 “No module named 'sklearn'” 时）
若 `pip list` 里已有 scikit-learn 但 `import sklearn` 失败，多半是安装不完整：
```bash
# 删除损坏的残留（仅 dist-info 无包文件时）
rm -rf $CONDA_PREFIX/lib/python3.11/site-packages/scikit_learn* $CONDA_PREFIX/lib/python3.11/site-packages/sklearn
pip install scikit-learn
```

### 1.3 threadpoolctl 重装（仅当 “No module named 'threadpoolctl'” 时）
若 pip 显示已安装但 Python 找不到，常是只有 dist-info 没有包代码：
```bash
pip install --force-reinstall threadpoolctl
```

## 2. 当前 ood 环境建议版本（已验证可跑 MOGB）

| 包名           | 建议版本   |
|----------------|------------|
| numpy          | 1.26.x, &lt;2 |
| scipy          | 1.11–1.14（如 1.14.1） |
| scikit-learn   | 任意当前稳定版（需完整安装） |
| threadpoolctl  | 任意当前稳定版（需完整安装） |
| torch          | 2.x        |
| transformers   | 5.x        |

## 3. 一键检查（在 ood 环境下运行）

```bash
conda activate ood
python -c "
import numpy; print('numpy', numpy.__version__)
import scipy; print('scipy', scipy.__version__)
import sklearn; print('sklearn ok')
import threadpoolctl; print('threadpoolctl ok')
from transformers import BertModel; print('transformers ok')
print('All imports OK.')
"
```

若上述无报错，在项目目录下运行：
```bash
cd /home/Alex/reproduce_MOGB
python MOGB.py --dataset stackoverflow --known_cls_ratio 0.25 --labeled_ratio 1.0 --seed 0 \
  --freeze_bert_parameters --bert_model /home/Alex/uncased_L-12_H-768_A-12 --gpu_id 0 \
  --train_batch_size 32 --eval_batch_size 64 --wait_patient 2 --num_train_epochs 2
```

## 4. 项目代码侧已做兼容（与本次环境修复一并生效）

- `util.py`: `AdamW` 改为从 `torch.optim` 导入
- `pretrain.py`: 去掉 `AdamW(correct_bias=True)` 的 `correct_bias` 参数
- `model.py`: 为 `BertForModel` 增加 `all_tied_weights_keys` 以兼容新版 transformers
