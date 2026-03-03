# ood 虚拟环境配置说明

用于在 **ood** 环境中运行本项目 `/home/david/reproduce_MOGB1`。

## 一、用 environment.yaml 配置（推荐）

在 **未运行其他 conda 命令** 的情况下，在终端执行：

```bash
cd /home/david
conda env update -f environment.yaml -n ood --prune
```

若环境尚未创建，则先创建：

```bash
conda env create -f /home/david/environment.yaml -n ood
```

若出现 `LockError: Failed to acquire lock`，请关闭其他占用 conda 的终端/IDE 后重试。

---

## 二、验证环境

激活环境并检查依赖是否能正常导入：

```bash
conda activate ood
python -c "
import numpy; print('numpy', numpy.__version__)
import scipy; print('scipy', scipy.__version__)
import sklearn; print('sklearn ok')
import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
from transformers import BertModel; print('transformers ok')
import matplotlib; print('matplotlib ok')
print('All imports OK.')
"
```

无报错即表示环境可用。

---

## 三、运行项目

BERT 模型默认路径为 `/gpu_data/uncased_L-12_H-768_A-12`（已在 `run.sh` 中配置）。

```bash
conda activate ood
cd /home/david/reproduce_MOGB1
bash run.sh
```

或指定参数运行（示例）：

```bash
python MOGB.py --dataset stackoverflow --known_cls_ratio 0.25 --labeled_ratio 1.0 --seed 0 \
  --freeze_bert_parameters --bert_model /gpu_data/uncased_L-12_H-768_A-12 --gpu_id 0 \
  --train_batch_size 32 --eval_batch_size 64 --wait_patient 2 --num_train_epochs 2
```

---

## 四、常见问题

| 问题 | 处理 |
|------|------|
| `ImportError: cannot import name '_spropack'` | 使用 `environment.yaml` 中 scipy 1.11~1.14 约束；或 `pip install "scipy>=1.11,<1.15"` |
| `No module named 'sklearn'` | `pip install --force-reinstall scikit-learn` |
| `No module named 'threadpoolctl'` | `pip install --force-reinstall threadpoolctl` |
| BERT 路径不同 | 设置环境变量：`BERT_MODEL=/你的路径 bash run.sh` |
| `nvidia-fabricmanager: command not found` | 580 驱动下可执行文件名为 **`nv-fabricmanager`**。手动启动：`sudo /usr/bin/nv-fabricmanager -d`；或用脚本：`sudo /usr/bin/nvidia-fabricmanager-start.sh --mode start`。仅 NVSwitch 多卡拓扑才需要此服务。 |
| **CUDA Error 802** / `torch.cuda.is_available()` 为 False | 若本机**无 NVSwitch**（见下）：先禁用 Fabric Manager 服务，802 需从容器/节点 GPU 可见性、驱动加载等排查。若有 NVSwitch 且 FM 能启动，再试 Python。 |
| **nvidia-fabricmanager.service failed** / `fabric manager don't have access permissions to directory /var/run/nvidia-fabricmanager` | 启动前需保证该目录存在且可写。在项目根目录已提供 `nvidia-fabricmanager-ensure-dir.conf`，按文件内注释用 sudo 复制到 `/etc/systemd/system/nvidia-fabricmanager.service.d/` 后执行 `sudo systemctl daemon-reload` 再启动服务。 |
| **nvidia-fabricmanager.service failed** / `NV_WARN_NOTHING_TO_DO` / `request to query NVSwitch device information ... failed` | 表示本机**没有 NVSwitch 设备**，Fabric Manager 不适用且会一直失败。**处理**：禁用服务即可，不影响单卡或普通多卡使用：`sudo systemctl stop nvidia-fabricmanager.service`，`sudo systemctl mask nvidia-fabricmanager.service`。 |

### CUDA 802 仍存在时（无 NVSwitch 环境）

在同一终端依次执行，用于排查：

```bash
nvidia-smi
ls -la /dev/nvidia*
groups
```

- 若 **nvidia-smi 报错或无 GPU**：当前环境很可能没有 GPU（如登录节点、未分配 GPU 的作业）。需在**分配了 GPU 的节点/作业**里再试（如 `srun --gres=gpu:1 ...` 或向管理员确认带 GPU 的节点）。
- 若 **nvidia-smi 正常** 但 **/dev/nvidia\*** 不存在或无读权限：可能是权限或驱动在该会话不可见，可尝试 `sudo nvidia-smi -pm 1` 或确认当前用户在 `video`/`render` 组。
- 若在**容器**中运行：需保证启动时已挂载 GPU（如 Docker `--gpus all`），并在容器内执行上述命令确认。
- 可运行项目内诊断脚本（与报错时同一终端）：`bash reproduce_MOGB1/check_cuda_802.sh`，根据输出判断是 nvidia-smi 不可用还是仅 Python 不可用。
- 若 **nvidia-smi 正常** 但 PyTorch 仍 802：先确认是否权限导致——在同一终端执行 `sudo python -c "import torch; print(torch.cuda.is_available())"`（需先 `conda activate ood`）。若此处为 **True**，多半是 `/dev/nvidia-caps/` 仅 root 可读导致，可用项目内 `70-nvidia-caps-permissions.rules` 修复。
- 若 **连 sudo 运行 Python 仍 802**：多半是驱动在等 Fabric 初始化。执行 `nvidia-smi -q | grep -A5 Fabric`，若看到 **Fabric State: In Progress**，表示驱动在等 Fabric Manager 完成，而本机 FM 无法启动（NV_WARN_NOTHING_TO_DO），故 CUDA 一直不可用。**可尝试**：① 先执行 `sudo nvidia-smi -r` 重置 GPU，再试 `python -c "import torch; print(torch.cuda.is_available())"`；② 联系管理员确认该机型/拓扑下是否有可用的 Fabric Manager 或驱动版本（如仅计算、不依赖 Fabric 的安装方式），或是否需更换/升级驱动与 FM 组合。
- **nvidia-smi -r 报 “In use by another client” / “could not be reset”**：先释放占用 GPU 的进程再重置。例如：`sudo systemctl stop nvidia-persistenced`，若有 `cloud-monitor` 等监控服务也占用 GPU 则临时停止（如 `sudo systemctl stop <服务名>`），再执行 `sudo nvidia-smi -r`，成功后执行 `sudo systemctl start nvidia-persistenced`（及前述监控服务）。勿用 `fuser -k` 强杀，否则 systemd 可能立即重启服务导致仍无法重置。

更多细节见项目内 `env_fix_ood.md`。
