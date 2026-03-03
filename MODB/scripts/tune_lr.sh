#!/usr/bin/env bash
# 自动调参脚本：主要调 lr / lr2 / lr3，参数会写入 outputs/results/results.csv
# 用法: bash scripts/tune_lr.sh
# 快速少组合: QUICK=1 bash scripts/tune_lr.sh
set -e
cd "$(dirname "$0")/.."

BERT_MODEL="${BERT_MODEL:-bert-base-uncased}"
DATASET="${DATASET:-banking}"
KNOWN_RATIO="${KNOWN_RATIO:-0.25}"
SEED="${SEED:-0}"

# 学习率候选（可改）
# lr: BERT 学习率  lr2: 粒球/分类头  lr3: 边界损失
if [ "${QUICK}" = "1" ]; then
  LRS=(2e-5)
  LR2S=(1e-4)
  LR3S=(1e-4)
else
  LRS=(2e-5 3e-5 5e-5)
  LR2S=(5e-5 1e-4 2e-4)
  LR3S=(5e-5 1e-4 2e-4)
fi

total=$((${#LRS[@]} * ${#LR2S[@]} * ${#LR3S[@]}))
n=0
for lr in "${LRS[@]}"; do
  for lr2 in "${LR2S[@]}"; do
    for lr3 in "${LR3S[@]}"; do
      n=$((n + 1))
      echo "========== [$n/$total] lr=$lr lr2=$lr2 lr3=$lr3 =========="
      python main.py \
        --data_dir data \
        --dataset "$DATASET" \
        --known_cls_ratio "$KNOWN_RATIO" \
        --labeled_ratio 1.0 \
        --seed "$SEED" \
        --freeze_bert_parameters \
        --bert_model "$BERT_MODEL" \
        --gpu_id 0 \
        --lr "$lr" \
        --lr2 "$lr2" \
        --lr3 "$lr3" \
        --train_batch_size 128 \
        --eval_batch_size 128 \
        --wait_patient 10 \
        --pretrain_dir outputs/saved \
        --save_results_path outputs/results \
        --adaptive_boundary_epochs 20 \
        --beta 0.1 \
        --ood
    done
  done
done
echo "Done. Results saved to outputs/results/results.csv"
