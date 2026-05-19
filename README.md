# SecurePath: 安全感知的思维链推理路径选择方法

> 基于对比学习的代码安全评估方法，应用于LLM代码生成任务的路径选择

## 项目简介

SecurePath是一个两阶段框架，用于在LLM代码生成任务中选择安全的代码输出：

1. **阶段一**：在漏洞代码数据集上训练安全感知编码器
2. **阶段二**：将编码器集成到CoT推理框架，从多条候选代码中选择最安全的版本

## 技术路线

```
阶段一：安全感知编码器训练
├── 数据：漏洞代码(func_before) vs 修复代码(func_after)
├── 方法：对比学习(CodeBERT + InfoNCE Loss)
└── 输出：安全感知编码器 + 漏洞原型/安全原型

阶段二：CoT路径安全选择
├── 输入：用户代码生成请求
├── LLM采样：生成K条推理路径，每条输出一段代码
├── 安全评估：编码器评估每段代码的安全性
└── 输出：选择安全得分最高的代码
```

## 项目结构

```
SecurePath/
├── configs/              # 配置文件
├── data/                 # 数据处理
│   ├── processed/        # 处理后数据
│   └── dataset.py        # 数据加载类
├── models/               # 模型定义
├── scripts/              # 训练和评估脚本
│   ├── preprocess.py     # 数据预处理
│   └── train_encoder.py  # 编码器训练
├── outputs/              # 输出结果
└── README.md
```

## 数据集

使用MSR_data_cleaned数据集，包含188,636个C/C++函数，涵盖6种内存安全漏洞类型：

| CWE | 漏洞类型 |
|-----|----------|
| CWE-119 | 缓冲区溢出 |
| CWE-416 | Use After Free |
| CWE-125 | 越界读取 |
| CWE-476 | 空指针解引用 |
| CWE-190 | 整数溢出 |
| CWE-787 | 越界写入 |

## 快速开始

```bash
# 安装依赖
pip install torch transformers datasets

# 数据预处理
python scripts/preprocess.py

# 训练编码器
python scripts/train_encoder.py
```

## 参考文献

- CodeBERT: [Feng et al., EMNLP 2020](https://arxiv.org/abs/2002.08155)
- GraphCodeBERT: [Guo et al., ICLR 2021](https://arxiv.org/abs/2009.08366)
- SimCSE: [Gao et al., EMNLP 2021](https://arxiv.org/abs/2104.08821)

## License

MIT License
