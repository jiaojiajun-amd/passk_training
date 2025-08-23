# Pass@1 和 Pass@k 验证指南

本指南介绍如何在VERL的PPO训练过程中同时验证pass@1和pass@k指标。

## 概述

在强化学习训练中，pass@k是一个重要的评估指标，表示在k个生成的回答中至少有一个正确答案的概率。VERL框架支持同时计算多个k值的pass@k指标。

## 配置方法

### 1. 设置验证采样参数

在配置文件中，需要设置`actor_rollout_ref.rollout.val_kwargs`参数：

```yaml
actor_rollout_ref:
  rollout:
    val_kwargs:
      # 设置n为最大k值，例如要计算pass@1, pass@5, pass@10，则设置n=10
      n: 10
      
      # 采样参数
      top_k: -1
      top_p: 1.0
      temperature: 0.8  # 使用temperature > 0来生成多样化的回答
      do_sample: True   # 启用采样以获得多样化的回答
```

### 2. 验证频率设置

设置验证频率：

```yaml
trainer:
  test_freq: 10  # 每10步验证一次
  val_before_train: True  # 训练前先验证
```

## 工作原理

### 验证流程

1. **生成多样化回答**：对于每个验证样本，模型会生成n个不同的回答
2. **计算得分**：使用奖励函数计算每个回答的得分
3. **计算pass@k**：对于每个k值，检查前k个最高得分的回答中是否有得分≥1.0的
4. **聚合指标**：计算所有样本的平均pass@k值

### 自动计算的指标

系统会自动计算以下指标：
- `val-core/{data_source}/pass_at_k/pass@1`
- `val-core/{data_source}/pass_at_k/pass@5` (如果n≥5)
- `val-core/{data_source}/pass_at_k/pass@10` (如果n≥10)

## 使用示例

### 基本配置

```yaml
# 配置文件示例
actor_rollout_ref:
  rollout:
    val_kwargs:
      n: 10  # 生成10个回答用于验证
      temperature: 0.8
      do_sample: True

trainer:
  test_freq: 10
  val_before_train: True
```

### 运行训练

```bash
python verl/trainer/main_ppo.py \
  --config-path examples/ppo_trainer_pass_at_k_config.yaml \
  actor_rollout_ref.model.path=/path/to/your/model \
  data.train_files=["/path/to/train.parquet"] \
  data.val_files=["/path/to/val.parquet"]
```

## 注意事项

### 1. 内存使用

生成多个回答会增加内存使用量，请确保有足够的GPU内存。

### 2. 验证时间

生成更多回答会增加验证时间，建议根据实际需求调整`n`值。

### 3. 奖励函数

确保你的奖励函数能够正确区分正确和错误的回答：
- 正确答案应该返回得分≥1.0
- 错误答案应该返回得分<1.0

### 4. 采样参数

为了获得多样化的回答：
- 设置`temperature > 0`
- 启用`do_sample: True`
- 可以调整`top_p`和`top_k`参数

## 监控指标

在训练过程中，你可以在日志中看到类似以下的指标：

```
val-core/gsm8k/pass_at_k/pass@1: 0.45
val-core/gsm8k/pass_at_k/pass@5: 0.78
val-core/gsm8k/pass_at_k/pass@10: 0.89
```

这些指标表示：
- pass@1: 45%的问题在第一个回答中得到了正确答案
- pass@5: 78%的问题在前5个回答中至少有一个正确答案
- pass@10: 89%的问题在前10个回答中至少有一个正确答案

## 故障排除

### 问题1: 所有pass@k值都相同

**原因**: 可能只生成了一个回答，或者所有回答的得分都相同。

**解决方案**: 
- 检查`val_kwargs.n`是否大于1
- 确保`do_sample: True`和`temperature > 0`
- 检查奖励函数是否正确区分不同质量的回答

### 问题2: 验证时间过长

**原因**: 生成了太多回答。

**解决方案**: 
- 减少`val_kwargs.n`的值
- 增加`test_freq`的值，减少验证频率

### 问题3: 内存不足

**原因**: 同时生成太多回答导致GPU内存不足。

**解决方案**: 
- 减少`val_kwargs.n`的值
- 减少`data.val_batch_size`的值
- 使用更小的模型或减少模型并行度

## 高级配置

### 自定义k值

如果需要计算特定的k值，可以修改代码中的`k_values`参数：

```python
# 在_validate方法中
k_values = [1, 3, 5, 10, 20]  # 自定义k值
```

### 多数据集支持

对于多个数据集，系统会为每个数据集分别计算pass@k指标：

```
val-core/gsm8k/pass_at_k/pass@1: 0.45
val-core/gsm8k/pass_at_k/pass@5: 0.78
val-core/math/pass_at_k/pass@1: 0.32
val-core/math/pass_at_k/pass@5: 0.65
```

## 总结

通过正确配置`val_kwargs.n`参数，VERL框架可以自动计算多个k值的pass@k指标，帮助你全面评估模型的性能。记住要设置合适的采样参数以获得多样化的回答，并根据你的硬件资源调整配置参数。
