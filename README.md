# bert-lora-imdb
Fine-tuning BERT with LoRA for sentiment analysis on IMDb movie reviews.

## 项目概述

本项目演示了如何利用LoRA技术高效微调预训练语言模型BERT，在IMDb电影评论数据集上完成情感分析任务（正面/负面）。通过LoRA，我们仅需训练极少量参数，即可使模型适配下游任务，显著降低计算和存储开销。

## 技术架构

### 模型结构
- **基座模型**: `google-bert/bert-base-uncased`
- **微调方法**: LoRA (Low-Rank Adaptation)
- **任务头**: 序列分类（2分类）

### LoRA 配置详情
本项目使用Hugging Face PEFT库实现LoRA，关键配置如下：
'''python
lora_config = LoraConfig(
r=8,
lora_alpha=32,
target_modules=["query", "key", "value"], # 注入到BERT的注意力模块
lora_dropout=0.05,
bias="none",
task_type="SEQ_CLS",
)
'''

### 数据处理流程
1. **分词**: 使用BERT对应的tokenizer，最大长度设置为256。
2. **标准化**: 进行填充（padding）和截断（truncation）。
3. **数据集划分**: 原始训练集进一步划分为训练集和验证集（90%/10%）。

## 训练细节

### 超参数设置
| 参数 | 值 | 说明 |
| :--- | :--- | :--- |
| **Batch Size** | 96 | 每设备训练批次大小 |
| **学习率** | 默认 (AdamW) | 优化器内建默认值 |
| **训练轮数** | 3 | |
| **最大序列长度** | 256 | |
| **随机种子** | 42 | 确保结果可复现 |

### 实验环境
- **框架**: Transformers, PEFT, PyTorch
- **硬件**: 单GPU（例如：NVIDIA T4 16GB）

## 结果与复现

### 模型性能
在IMDB测试集（25,000条评论）上的评估结果：
- **准确率 (Accuracy)**: `89.06%`

### 如何复现
安装好相应依赖运行即可

## 致谢
- 感谢Hugging Face提供Transformers和PEFT库。
- 感谢IMDb数据集的提供者。

## 许可
本项目为学习示例项目，欢迎讨论交流。
