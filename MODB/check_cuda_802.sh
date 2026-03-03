#!/bin/bash
# CUDA 802 诊断：在同一终端中运行，用于排查 torch.cuda.is_available() 为 False
# 用法: bash check_cuda_802.sh  或  conda activate ood && bash check_cuda_802.sh

echo "=== 1. nvidia-smi ==="
nvidia-smi 2>&1 || echo "nvidia-smi 失败"
echo ""
echo "=== 2. /dev/nvidia* ==="
ls -la /dev/nvidia* 2>&1 || echo "无 /dev/nvidia*"
echo ""
echo "=== 3. 当前用户 groups ==="
groups
echo ""
echo "=== 4. Python / PyTorch CUDA ==="
python -c "
import sys
print('python:', sys.executable)
import torch
print('torch:', torch.__version__)
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device_count:', torch.cuda.device_count())
else:
    print('(CUDA 不可用)')
" 2>&1
echo ""
echo "=== 5. 环境变量 ==="
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<未设置>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<未设置>}"
