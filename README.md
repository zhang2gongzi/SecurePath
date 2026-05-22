# SecurePath: 安全感知的思维链推理路径选择方法

> 基于对比学习的代码安全评估方法，应用于LLM代码生成任务的路径选择

## 当前状态

- ✅ **阶段一完成**：安全感知编码器训练，准确率 **85.7%**（CodeBERT + 对比学习）
- ✅ **原型已生成**：`vul_prototype.pt`（漏洞原型）、`safe_prototype.pt`（安全原型）
- ⏳ **阶段二进行中**：CoT路径安全选择实验

## 项目简介

SecurePath是一个两阶段框架，用于在LLM代码生成任务中选择安全的代码输出：

1. **阶段一** ✅：在漏洞代码数据集上训练安全感知编码器，构建漏洞/安全原型
2. **阶段二** ⏳：将编码器集成到CoT推理框架，从多条候选代码中选择最安全的版本

## 技术路线

```
阶段一 ✅：安全感知编码器训练
├── 数据：方案二数据池（不同函数构建漏洞/安全代码对）
├── 方法：对比学习(CodeBERT + InfoNCE Loss + 投影头 768→256)
├── 结果：准确率 85.7%（从52%提升）
└── 输出：best_model + vul_prototype.pt + safe_prototype.pt

阶段二 ⏳：CoT路径安全选择
├── 输入：用户代码生成请求
├── LLM采样：生成K条推理路径，每条输出一段代码
├── 安全评估：编码器 + 原型相似度评估安全性
└── 输出：选择安全得分最高的代码
```

## 项目结构

```
SecurePath/
├── SecureProtoCoT/
│   ├── configs/              # 配置文件
│   ├── data/                 # 数据处理
│   │   ├── processed/        # train_contrastive.csv, test_contrastive.csv
│   │   └── dataset.py        # 数据加载类
│   ├── scripts/              # 训练和评估脚本
│   │   ├── preprocess.py     # 数据预处理
│   │   ├── train_encoder.py  # 编码器训练（对比学习）
│   │   ├── build_prototypes.py # 原型构建
│   │   ├── train_classifier.py # 分类器训练（备选）
│   │   └── visualize_embeddings.py # 嵌入可视化
│   └── outputs/
│       └── models/
│           ├── best_model/   # 最佳编码器
│           ├── vul_prototype.pt
│           └── safe_prototype.pt
├── CoT安全性研究-完整方案.md
├── 实验执行清单.md
└── README.md
```

## 数据集

使用MSR_data_cleaned数据集，188,636个C/C++函数，方案二数据池：

| 数据池 | 来源 | 数量 |
|--------|------|------|
| 漏洞代码池 | vul=1 的 func_before | 7,901 |
| 安全代码池 | vul=0 的 func_before + vul=1 的 func_after | 118,988 |

## 快速开始

```bash
# 安装依赖
pip install torch transformers

# 数据预处理（已完成）
python SecureProtoCoT/scripts/preprocess.py

# 训练编码器（已完成，acc=85.7%）
python SecureProtoCoT/scripts/train_encoder.py

# 构建原型（已完成）
python SecureProtoCoT/scripts/build_prototypes.py
```

## 参考文献

- CodeBERT: [Feng et al., EMNLP 2020](https://arxiv.org/abs/2002.08155)
- GraphCodeBERT: [Guo et al., ICLR 2021](https://arxiv.org/abs/2009.08366)
- SimCSE: [Gao et al., EMNLP 2021](https://arxiv.org/abs/2104.08821)

## License

MIT License
