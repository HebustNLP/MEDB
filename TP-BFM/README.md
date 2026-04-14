## 目录说明

- `data/`
  - `dataset.py`：OOS 数据读取、标签映射、已知/未知类别划分
  - `seed_selector.py`：每类种子样本抽样
- `augmentation/`
  - `tuple_prompt.py`：元组提示构建
  - `sample_generator.py`：基于本地模型或 OpenAI API 生成增强样本
- `filtering/`
  - `soft_label.py`：软标签分布生成
  - `confidence_filter.py`：第一层过滤（置信度 + 标签一致性）
  - `pvi_filter.py`：第二层过滤（PVI）
- `open_intent_detection/`与EDB框架一致
  - `examples`：运行脚本
- `config.py`：统一参数配置