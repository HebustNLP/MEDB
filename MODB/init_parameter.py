# -*- coding: utf-8 -*-
"""
MOGB 多粒度粒球算法：全局参数定义。
纯度与球分裂过程均由以下参数控制，详见 PARAMS.md。
"""
import argparse


def init_model():
    parser = argparse.ArgumentParser(description="MOGB: 多粒度粒球 + 开放集识别")

    # ---------- 数据与任务 ----------
    parser.add_argument("--data_dir", type=str, default="data", help="数据根目录")
    parser.add_argument("--dataset", type=str, default="oos",
                        choices=["oos", "clinc", "dbpedia", "stackoverflow", "banking", "snips", "ATIS", "tc20"],
                        help="数据集名称")
    parser.add_argument("--known_cls_ratio", type=float, default=0.75, help="已知类占比 (0~1)")
    parser.add_argument("--labeled_ratio", type=float, default=1.0, help="训练时使用的标注比例 (0~1)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # ---------- 粒球与纯度（控制聚类/分裂） ----------
    # 分裂条件：当 纯度 < 对应 purity 且 样本数 > 对应 min_ball 时，对该球继续分裂
    parser.add_argument("--purity_train", type=float, default=0.9,
                        help="[分裂] 训练阶段粒球纯度阈值：低于此值且样本数>min_ball_train 则分裂")
    parser.add_argument("--min_ball_train", type=int, default=3,
                        help="[分裂] 训练阶段最小球规模：样本数大于此值才允许因低纯度而分裂")
    parser.add_argument("--purity_get_ball", type=float, default=0.9,
                        help="[分裂] 计算最终粒球时的纯度阈值：低于此值且样本数>min_ball_get_ball 则分裂")
    parser.add_argument("--min_ball_get_ball", type=int, default=3,
                        help="[分裂] 计算最终粒球时的最小球规模")

    # 选球条件：分裂完成后，只保留 纯度 > 对应 purity 且 样本数 >= 对应 min_ball 的球
    parser.add_argument("--purity_select_ball_train", type=float, default=0.1,
                        help="[选球] 训练阶段保留粒球的纯度下界（>=此值才保留，默认0.1即几乎全保留）")
    parser.add_argument("--min_ball_select_ball_train", type=int, default=1,
                        help="[选球] 训练阶段保留粒球的最小样本数")
    parser.add_argument("--purity_select_ball", type=float, default=0.8,
                        help="[选球] 最终粒球保留的纯度下界（>=此值才用于推理）")
    parser.add_argument("--min_ball_select_ball", type=int, default=2,
                        help="[选球] 最终粒球保留的最小样本数")

    # ---------- 模型与训练 ----------
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", help="BERT 预训练模型名或路径")
    parser.add_argument("--freeze_bert_parameters", action="store_true", help="是否冻结 BERT 大部分参数")
    parser.add_argument("--feat_dim", type=int, default=768, help="特征维度（与 BERT 输出一致）")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU 编号")
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--wait_patient", type=int, default=5, help="验证集无提升的容忍轮数，达此数则早停")
    parser.add_argument("--save_model", action="store_true", help="是否保存预训练权重")
    parser.add_argument("--pretrain_dir", type=str, default="saved", help="预训练模型保存目录")
    parser.add_argument("--save_results_path", type=str, default="results", help="测试结果与混淆矩阵保存目录")

    # 优化器与学习率
    parser.add_argument("--lr", type=float, default=2e-5, help="BERT 编码器学习率")
    parser.add_argument("--lr2", type=float, default=1e-4, help="粒球/分类头学习率")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, help="warmup 步数占比")

    # 动态边界（可选）
    parser.add_argument("--lr3", type=float, default=1e-4, help="自适应边界模块学习率")
    parser.add_argument("--adaptive_boundary_epochs", type=int, default=0,
                        help=">0 时在粒球基础上训练可学习边界；0 则使用固定半径")
    parser.add_argument("--beta", type=float, default=0.1, help="边界损失中 OOD 边际权重")
    parser.add_argument("--triangle", action="store_true", help="椭球是否使用三角参数化（节省参数）")
    parser.add_argument("--shape", type=str, default="ball", choices=["ball", "ellipse"],
                        help="决策边界形状：ball=球，ellipse=椭球")
    parser.add_argument("--ood", type=bool, default=True, help="边界损失是否包含 OOD 边际")

    return parser
