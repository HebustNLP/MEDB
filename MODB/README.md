## This is the code for paper "Multi-Granularity Open Intent Classification via Adaptive Granular-Ball Decision Boundary (MOGB)"

### 运行说明

1. **环境**：需要 Python 3.8+，并安装：`torch`、`transformers`、`pandas`、`scikit-learn`、`tqdm`、`matplotlib`。
   - 若出现 `numpy`/`accelerate` 相关报错，可先执行：`pip install "numpy<2"` 或使用新的虚拟环境。

2. **BERT 路径**：默认使用 `/home/Alex/uncased_L-12_H-768_A-12`。  
   - 可通过命令行参数 `--bert_model /path/to/bert` 覆盖，或设置环境变量：  
     `BERT_MODEL=/path/to/bert ./run.sh`

3. **运行**（在项目根目录下）：
   ```bash
   cd /home/Alex/reproduce_MOGB
   bash run.sh
   ```
   或直接指定参数：
   ```bash
   python MOGB.py --dataset stackoverflow --known_cls_ratio 0.25 --labeled_ratio 1.0 --seed 0 \
     --freeze_bert_parameters --bert_model /home/Alex/uncased_L-12_H-768_A-12 --gpu_id 0 \
     --train_batch_size 128 --eval_batch_size 128 --wait_patient 10
   ```
   注意：直接运行 `python MOGB.py` 时，当前工作目录需在 `reproduce_MOGB` 下，否则 `data_dir`/`pretrain_dir` 会指向错误路径。

