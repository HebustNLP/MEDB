
from pretrain import *
# from utils import util

from gb_test import ModelManager


if __name__ == '__main__':
    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    print('Parameters Initialization...')

    parser = init_model()
    args = parser.parse_args()
    data = Data(args)

    # 训练前打印当前使用 GPU 还是 CPU（gpu_id 仅在检测到 GPU 时生效，否则使用 CPU）
    if torch.cuda.is_available():
        print(f'当前使用: GPU (cuda:{args.gpu_id}, device={torch.cuda.get_device_name(0)})')
    else:
        print('当前使用: CPU（未检测到可用 GPU，run.sh 中的 gpu_id 仅在存在 GPU 时生效）')

    print('Training begin...')
    manager_p1 = PretrainModelManager(args, data)
    manager_p1.train(args, data)
    print('Training finished!')

    print('Calculate ball begin...')
    gb_centroids, gb_radii, gb_labels = manager_p1.calculate_granular_balls(args, data)
    print('Calculate ball finished!')

    # 动态自适应决策边界：在多粒度球基础上学习边界
    boundary_loss = None
    if getattr(args, 'adaptive_boundary_epochs', 0) > 0:
        print('Train adaptive boundary begin...')
        gb_centroids, gb_radii, gb_labels, boundary_loss = manager_p1.train_adaptive_boundary(
            args, data, gb_centroids, gb_radii, gb_labels)
        print('Train adaptive boundary finished!')

    manager = ModelManager(args, data, manager_p1.model)
    print('Evaluation begin...')
    manager.evaluation(args, data, gb_centroids, gb_radii, gb_labels, mode="test", boundary_loss=boundary_loss)
    print('Evaluation finished!')
    # util.summary_writer.close()