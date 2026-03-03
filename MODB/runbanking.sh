#!/usr/bin/env bash
# 进入项目目录，保证 data_dir='data' 和 pretrain_dir='models' 正确解析
cd "$(dirname "$0")"

# BERT 路径（可通过环境变量覆盖）
BERT_MODEL="${BERT_MODEL:-/home/Alex/uncased_L-12_H-768_A-12}"

for dataset in banking
do
    for known_cls_ratio in 0.25 0.5 0.75
    do
        for seed in 0 1 2 3 4
        do
            # 三个学习率：--lr 表示学习，--lr2 最近子质心/多粒度球，--lr3 边界损失（默认同 lr2）
            python MOGB.py \
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
                --adaptive_boundary_epochs 20 \
                --beta 0.1 \
                --ood
            # 仅用多粒度球（不学边界）时可加: --adaptive_boundary_epochs 0
        done
    done
done