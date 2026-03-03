#!/usr/bin/env bash
# 进入项目目录，保证 data_dir 和 pretrain_dir 正确解析
cd "$(dirname "$0")"

# BERT 路径（可通过环境变量 BERT_MODEL 覆盖）
BERT_MODEL="${BERT_MODEL:-/gpu_data/uncased_L-12_H-768_A-12}"

# ---------- 以下为可传入参数（按 init_parameter.py 列出，未写出的使用脚本内默认或 init 默认） ----------
# 数据与任务
#   --data_dir         数据根目录，默认 data
#   --dataset          数据集: oos|clinc|dbpedia|stackoverflow|banking|snips|ATIS|tc20
#   --known_cls_ratio  已知类占比 (0~1)
#   --labeled_ratio    训练标注比例 (0~1)
#   --seed             随机种子
# 粒球与纯度（分裂与选球，必须传入以显式控制）
#   --purity_train                 训练阶段分裂纯度阈值
#   --min_ball_train              训练阶段分裂最小球规模
#   --purity_get_ball             最终粒球分裂纯度阈值
#   --min_ball_get_ball           最终粒球分裂最小球规模
#   --purity_select_ball_train    训练阶段选球纯度下界
#   --min_ball_select_ball_train  训练阶段选球最小样本数
#   --purity_select_ball          最终选球纯度下界
#   --min_ball_select_ball        最终选球最小样本数
# 模型与训练
#   --bert_model              BERT 模型名或路径
#   --freeze_bert_parameters  是否冻结 BERT（flag）
#   --feat_dim                特征维度，默认 768
#   --gpu_id                  GPU 编号
#   --train_batch_size
#   --eval_batch_size
#   --num_train_epochs
#   --wait_patient            早停耐心轮数
#   --save_model              是否保存模型（flag）
#   --pretrain_dir            预训练保存目录
#   --save_results_path       结果保存目录
# 优化器
#   --lr                   BERT 学习率
#   --lr2                  粒球/分类头学习率
#   --warmup_proportion    warmup 比例
# 动态边界
#   --lr3                      边界模块学习率
#   --adaptive_boundary_epochs  边界训练轮数，0 表示不训练边界
#   --beta                     OOD 边际权重
#   --triangle                 椭球三角参数化（flag）
#   --shape                    ball|ellipse
#   --ood                      是否含 OOD 边际

for dataset in stackoverflow; do
    for known_cls_ratio in 0.5; do
        for seed in 0 ; do
            python MOGB.py \
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
                --pretrain_dir saved \
                --save_results_path results \
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