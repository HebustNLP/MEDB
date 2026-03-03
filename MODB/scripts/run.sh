#!/usr/bin/env bash
# 进入项目根目录
cd "$(dirname "$0")/.."

# BERT 路径（可通过环境变量 BERT_MODEL 覆盖）
BERT_MODEL="${BERT_MODEL:-bert-base-uncased}"

for dataset in stackoverflow; do
    for known_cls_ratio in 0.5; do
        for seed in 0 ; do
            python main.py \
                --data_dir data \
                --dataset "$dataset" \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio 1.0 \
                --seed $seed \
                --bert_model "$BERT_MODEL" \
                --freeze_bert_parameters \
                --gpu_id 0 \
                --feat_dim 768 \
                --train_batch_size 128 \
                --eval_batch_size 128 \
                --num_train_epochs 15 \
                --wait_patient 10 \
                --pretrain_dir outputs/saved \
                --save_results_path outputs/results \
                --lr 2e-2 \
                --lr2 0.0006 \
                --warmup_proportion 0.1 \
                --lr3 5e-4 \
                --adaptive_boundary_epochs 10 \
                --beta 0.1 \
                --shape ball \
                --purity_train 0.9 \
                --min_ball_train 3 \
                --purity_get_ball 0.9 \
                --min_ball_get_ball 3 \
                --purity_select_ball_train 0.1 \
                --min_ball_select_ball_train 1 \
                --purity_select_ball 0.8 \
                --min_ball_select_ball 2 \
                --ood OOD
        done
    done
done
