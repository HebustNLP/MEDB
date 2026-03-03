#!/usr/bin/env bash
# 进入项目根目录
cd "$(dirname "$0")/.."

# BERT 路径（可通过环境变量 BERT_MODEL 覆盖）
BERT_MODEL="${BERT_MODEL:-bert-base-uncased}"

for dataset in oos; do
    for known_cls_ratio in 0.25 0.5 0.75; do
        for seed in 0 1 2 3 4; do
            python main.py \
                --data_dir data \
                --dataset $dataset \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio 1.0 \
                --seed $seed \
                --freeze_bert_parameters \
                --bert_model "$BERT_MODEL" \
                --gpu_id 0 \
                --lr 2e-5 \
                --lr2 0.0001 \
                --lr3 5e-5 \
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
