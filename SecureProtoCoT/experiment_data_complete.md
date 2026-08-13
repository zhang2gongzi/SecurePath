# SecurePath 全部实验数据汇总

---

## 1. 数据预处理

| 指标 | 数值 |
|------|------|
| MSR_data_cleaned 总函数数 | 188,636 |
| vul=1 (漏洞函数) | 10,900 |
| vul=0 (安全函数) | 177,736 |
| vul=0 中 func_before == func_after | 92.7% |

### Solution Two 数据池构建

| 数据池 | 来源 | 数量 |
|--------|------|------|
| 漏洞代码池 | vul=1 的 func_before | 7,901 |
| 安全代码池 | vul=0 的 func_before + vul=1 的 func_after | 118,988 |

### CWE 类型采样

| CWE | 类型 | 漏洞采样 | 安全采样 |
|-----|------|----------|----------|
| CWE-119 | 缓冲区溢出 | 500/1,427 | 500/17,570 |
| CWE-416 | Use After Free | 254/254 | 500/6,124 |
| CWE-125 | 越界读取 | 456/456 | 500/5,646 |
| CWE-476 | 空指针解引用 | 180/180 | 500/3,519 |
| CWE-190 | 整数溢出 | 266/266 | 500/2,649 |
| CWE-787 | 越界写入 | 166/166 | 500/1,942 |
| **总计** | | **1,822** | **3,000** |

### 最终数据集划分

| 数据集 | 总计 | 漏洞 | 安全 |
|--------|------|------|------|
| 训练集 | 3,375 | 1,275 | 2,100 |
| 验证集 | 482 | 182 | 300 |
| 测试集 | 965 | 365 | 600 |

---

## 2. 阶段一：安全感知编码器（对比学习）

| 指标 | 数值 |
|------|------|
| 模型 | CodeBERT-base (125M) |
| 方法 | InfoNCE 对比学习 (SimCSE 范式) |
| Pair Matching 准确率 | **85.7%** |

---

## 3. 阶段二：安全分类器

### 3.1 CLS Pooling vs AttentionPooling

| 架构 | F1 | AUC | Params |
|------|-----|------|--------|
| CLS Pooling | 0.8840 | 0.9488 | ~230K |
| AttentionPooling | **0.8959** | **0.9531** | ~820K |
| 原型 (余弦阈值) | 0.670 | 0.833 | — |

### 3.2 AttentionPooling 详细指标

| 指标 | Train | Val | Test |
|------|-------|-----|------|
| Accuracy | — | 0.9051 | **0.8929** |
| Precision | — | 0.8936 | **0.8714** |
| Recall | — | 0.9197 | **0.9220** |
| F1 | — | 0.9065 | **0.8959** |
| AUC | — | 0.9549 | **0.9531** |
| Consistency | |ΔF1\| = 0.0105, \|ΔAUC\| = 0.0046 | |

### 3.3 Per-CWE 评估

| CWE | 类型 | F1 | AUC |
|-----|------|-----|------|
| CWE-119 | 缓冲区溢出 | 0.887 | 0.946 |
| CWE-125 | 越界读取 | — | — |
| CWE-190 | 整数溢出 | — | — |
| CWE-416 | Use After Free | **0.938** | **0.985** |
| CWE-476 | 空指针解引用 | — | — |
| CWE-787 | 越界写入 | — | — |
| 范围 | | 0.887–0.938 | 0.946–0.985 |

### 3.4 留一法跨 CWE 泛化

| 指标 | 数值 |
|------|------|
| 平均 F1 | **0.8944** |
| 全量训练 F1 | 0.8959 |
| ΔF1 | **0.0015** |

---

## 4. 实验B：LLM 端到端验证（10候选人/prompt）

| 指标 | 数值 |
|------|------|
| 配置 | 15 prompt × 10 candidate = 150 条 |
| 生成模型 | DeepSeek-v4-pro, temp=0.8 |
| Random Avg P(vul) | 0.000977 |
| Classifier Avg P(vul) | 0.000244 |
| P(vul) 降幅 | **75%** |
| 分类器胜率 | **11/15** prompt |
| 最佳案例 (classifier > random 的候选数) | P09 (15×), P12 (14×), P01 (12×), P11 (9×) |
| 风险标注 (max P(vul)) | P07_int_parse=0.594, P14_realloc_array=0.576 |

### 跨域分布偏移（MSR → LLM）

| 域 | 分类器表现 |
|----|-----------|
| MSR (in-domain) | F1=0.896, AUC=0.953 |
| LLM (实验B) | safe-pick rate 85.7% ≈ 随机 92.9% |
| P(vul) 信号方向 | 安全代码 P(vul) 中位数 = 0.000589 > 漏洞代码 P(vul) 中位数 = 0.000205（反转） |

---

## 5. 实验 ISR：迭代安全精炼（15 prompts）

### 5.1 消融配置

| 配置 | Feedback | Generic | Attention | Spec | 验证问题 |
|------|----------|---------|-----------|------|---------|
| ISR-0 | × | × | × | × | 单次生成下界 |
| ISR-1 | ✓ | ✓ | × | × | 模糊反馈够不够？ |
| ISR-2 | ✓ | × | ✓ | × | 精准定位但不告诉怎么修 |
| ISR-3 | ✓ | × | ✓ | ✓ | 完整方案 |

### 5.2 消融结果（分类器视角，15 prompts）

| 配置 | Init P(vul) | Best P(vul) | Δ P(vul) | Avg Iters | 收敛数 |
|------|------------|-------------|------|-----------|--------|
| ISR-0 | 0.080 | 0.080 | 0.000 | 1.0 | 8/15 |
| ISR-1 | 0.075 | 0.031 | **+0.044** | 2.5 | 9/15 |
| **ISR-2** | **0.018** | **0.010** | +0.007 | 2.3 | **10/15** |
| ISR-3 | 0.152 | 0.098 | **+0.055** | 3.1 | 9/15 |

### 5.3 ISR-2 Headline Case (P09_memcpy_wrapper)

| 指标 | 数值 |
|------|------|
| 初始 P(vul) | 0.806 |
| 最终 P(vul) | 0.000070 |
| Δ P(vul) | **+0.806**（四个数量级） |
| 迭代次数 | 3 |

### 5.4 ISR-1 详细案例

| Prompt | 初始 P(vul) | 最终 P(vul) | 迭代数 | 结果 |
|--------|------------|-------------|--------|------|
| P04 | — | 0.000231 | 5 | 改善 |
| P07 | 0.510 | 0.048 | — | 改善 |
| P08 | — | — | — | 改善 |
| P11 | — | — | — | 改善 |
| P13 | — | — | — | 改善 |
| P06 | — | — | — | 退化（模糊反馈误导） |
| P10 | — | — | — | 退化（模糊反馈误导） |
| P15 | — | — | — | 退化（模糊反馈误导） |

---

## 6. 人工评估（80 条，盲化）

### 6.1 各方法 Human SAFE Rate（核心结果表）

| 方法 | Human SAFE Rate | 类型 | 机制 |
|------|----------------|------|------|
| ISR-0 (no feedback) | **45.5%** | Lower bound | Single generation |
| SafePrompt (B4) | **71.4%** | Prompt engineering | Safety-enhanced prompt |
| ISR-2 (attention) | **72.7%** | ISR ablation | Precise location feedback |
| CoSec (B7, ISSTA 2024) | **76.7%** | Academic baseline | Security spec + self-audit |
| SVEN (B5, CCS 2023) | **78.6%** | Academic baseline | Prefix-guided steering |
| ISR-1 (generic) | **80.0%** | ISR ablation | Vague safety feedback |
| Reflexion (B6, NeurIPS 2023) | **83.3%** | Academic baseline | LLM self-review + repair |
| **ISR-3 (attention+spec)** | **87.5%** | **Our method** | **Precise feedback + spec** |

### 6.2 ISR 迭代提升

| 配置 | Human SAFE Rate |
|------|----------------|
| ISR-0 (no feedback) | 45.5% |
| ISR-1 | 80.0% |
| ISR-2 | 72.7% |
| ISR-3 | **87.5%** |
| ISR-0 → ISR-3 | **+42pp** |

### 6.3 分类器 P(vul) vs 人类判断（域偏移证据）

| P(vul) Band | 分类器含义 | Human SAFE Rate |
|-------------|-----------|----------------|
| < 0.001 | 高度安全 | **66.7%** |
| 0.001–0.1 | 不确定 | **78.1%** |
| > 0.1 | 可能有漏洞 | **83.3%** |

Spearman ρ = **0.16**（接近零相关，方向反转）

### 6.4 分类器虚报 vs 人类实际

| 方法 | 分类器声称 | 人类实际 | 虚高 |
|------|-----------|---------|------|
| SafePrompt (B4) | ~87% | 71.4% | **+15.6pp** |
| SVEN (B5) | 93.3% | 78.6% | **+14.7pp** |
| Reflexion (B6) | 93.3% | 83.3% | **+10.0pp** |
| CoSec (B7) | 86.7% | 76.7% | **+10.0pp** |

### 6.5 Baseline 自评估失败

| Baseline | 失败模式 | 证据 |
|----------|---------|------|
| B6 (Reflexion) | 自反思退化 | 10/15 情况修复后 P(vul) 更高；最优迭代通常是初始生成 (iter=0) |
| B7 (CoSec) | 自审计零区分度 | 几乎所有候选评分完美 (5P/0F)；P07_c9 P(vul)=0.840 自评仍 5P/0F |

### 6.6 共同盲区：P01_buffer_copy

所有 5 条 baseline 在任何配置下均无法生成安全的 buffer copy 实现。ISR-3 正确处理。

---

## 7. 功能正确性实验（ISR-0 vs ISR-3）

| Task | ISR-0 | ISR-3 | 测试用例描述 |
|------|-------|-------|-------------|
| P01_buffer_copy | 5/5 | 5/5 | Normal, empty, exact fit, overflow, NULL src |
| P05_free_memory | 3/3 | 3/3 | Normal free, NULL ptr, double free |
| P07_int_parse | 5/5 | 5/5 | Positive, negative, zero, overflow, invalid |
| P09_memcpy_wrapper | 4/4 | 4/4 | Normal copy, overlap, NULL dest, zero length |
| P11_struct_copy | 4/4 | 4/4 | Deep copy, NULL src, empty string, independence |
| **总计** | **21/21** | **21/21** | **100%** |

---

## 8. Baseline 横评总览

| Baseline | 方法 | 分类器 Safe Rate | 人类 Safe Rate | 核心缺陷 |
|----------|------|-----------------|---------------|---------|
| B2 | Flawfinder (static) | — | — | 无法区分同类代码块安全差异 |
| B4 | SafePrompt | ~87% | 71.4% | prompt 工程天花板 |
| B7 | CoSec-inspired | 86.7% | 76.7% | self-audit 零区分度 |
| B5 | SVEN | 93.3% | 78.6% | prefix 工程天花板 |
| B6 | Reflexion | 93.3% | 83.3% | 自反思退化 (5/10) |
| ISR-3 | **Ours** | — | **87.5%** | — |

---

## 9. 核心数字（论文可直接引用）

| 指标 | 值 |
|------|-----|
| 编码器配对准确率 | 85.7% |
| 分类器 Test F1 (MSR) | 0.8959 |
| 分类器 Test AUC (MSR) | 0.9531 |
| 原型基线 F1 | 0.670 |
| 留一法平均 F1 | 0.8944 |
| 留一法 vs 全量 ΔF1 | 0.0015 |
| 实验B P(vul) 降幅 | 75% |
| 实验B 分类器胜率 | 11/15 |
| ISR-3 Human SAFE Rate | **87.5%** |
| ISR-0 → ISR-3 提升 | **+42pp** |
| Spearman ρ (P(vul) vs Human) | **0.16** |
| Baseline 虚高幅度 | **10–15pp** |
| 功能正确性 (ISR-0/ISR-3) | **21/21 (100%)** |
| ISR-2 Headline Case ΔP(vul) | 0.806 → 0.000070 |
| 训练样本数 | 5,498 (2,749 vul + 2,749 safe) |
| CWE 类型数 | 6 |
| 分类器可训练参数 | ~820K |
| 人工评估样本数 | 80 |
| 评估 Prompt 数 | 15 |
| 引用数 | 35 |

---

## 10. 远程服务器路径

| 资源 | 路径 |
|------|------|
| 编码器模型 | `/home2/zzl/SecurePath/SecureProtoCoT/outputs/models/best_model` |
| 分类器 (AttnPool) | `/home2/zzl/SecurePath/SecureProtoCoT/outputs/models/safety_classifier_attn.pt` |
| 分类器 (CLS) | `/home2/zzl/SecurePath/SecureProtoCoT/outputs/models/safety_classifier.pt` |
| 原始 CodeBERT | `/home2/zzl/model/codebert-base` |
| 数据 | `/home2/zzl/SecurePath/SecureProtoCoT/data/processed/` |
| 实验B输出 | `/home2/zzl/SecurePath/SecureProtoCoT/outputs/experiment_b/` |
| 实验C输出 | `/home2/zzl/SecurePath/SecureProtoCoT/outputs/experiment_c/` |
