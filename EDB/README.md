# EDB

EDB 是面向文本开放意图识别的开放意图检测方法。

## 简介

本项目基于 TEXTOIR 框架，实现 EDB 方法用于开放意图检测任务。开放意图检测旨在识别 n 类已知意图，并检测一类开放意图。

## 快速开始

1. 创建 Python 环境（版本 >= 3.6）
```
conda create --name edb python=3.6
conda activate edb
```

2. 安装 PyTorch（以 Cuda 11.0 为例）
```
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch -c conda-forge  
```

3. 克隆项目并进入开放意图检测目录
```
cd EDB
cd open_intent_detection
```

4. 安装依赖
```
pip install -r requirements.txt
```

5. 运行 EDB 示例（EDB 方法对应 `run_EliDecide.sh`）
```
sh examples/run_EliDecide.sh
```

* 若无法从 HuggingFace transformers 直接下载预训练模型，需自行下载。可参考 TEXTOIR 提供的 [百度网盘链接](https://pan.baidu.com/s/1k1zxK4xh0UyPhOU_-oPlow)（提取码: v8tk）。

## 基准数据集

| 数据集 | 说明 |
| :---: | :---: |
| [BANKING](./data/banking) | 银行领域意图 |
| [OOS](./data/oos) / [CLINC150](./data/clinc) | 通用对话意图 |
| [StackOverflow](./data/stackoverflow) | 技术问答意图 |

## 引用

若使用本代码，请引用相关论文。
