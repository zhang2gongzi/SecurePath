# SecurePath 实验C重构计划：分类器驱动的安全生成

## 背景

原实验B（纯post-hoc过滤）和实验C（prompt engineering + ensemble）的学术含金量不足。根本问题：
1. 安全信号始终在生成之后介入，未影响LLM的生成决策
2. 分类器训练在MSR真实代码，对LLM生成代码存在分布偏移，多信号ensemble只是弱信号叠加

### 关键发现：MSR → LLM 跨域分布偏移（Cross-Domain Distribution Shift）

实验B 暴露了一个 solid 发现，可以作为论文核心 motivation：

| 域 | 分类器表现 | 证据 |
|----|-----------|------|
| MSR 源域（训练数据同分布） | F1=0.896, AUC=0.953 | evaluate_safety.py test set |
| LLM 生成域（DeepSeek-v4-pro） | safe-pick rate = 85.7% ≈ 随机 92.9% | human_eval.csv × classifier |

**根因**：P(vul) 信号在 LLM 生成代码上失去区分力：
- 安全代码中位数 P(vul) = 0.000589
- 漏洞代码中位数 P(vul) = 0.000205
- 信号方向反转——分类器认为安全的代码实际上更危险

**论文以这个发现作为 motivation 的价值**：
1. 诚实：展示了方法的局限性，不是硬编故事
2. 有深度：映射回 MSR 数据和 LLM 数据之间的结构性差异（token 分布、漏洞模式、代码风格）
3. 引出后续方法：因为纯后过滤不可靠，所以需要 ISR（让分类器反馈进入生成过程）和 DPO（从参数层面解决）

**论文核心描述句**：
> "分类器与随机选择在安全选择准确率上无统计显著差异（85.7% vs 92.9%, n=14），这表明 MSR 域训练的分类器在 LLM 生成代码上未能提供有效的安全信号——这一分布偏移问题正是本文要解决的。"

**可做的分析** (强化这个发现)：
- 用 t-SNE 可视化 MSR 代码 vs LLM 生成代码在 256 维对比空间中的分布
- 统计两个域代码的 token 长度、关键词频率、复杂度差异
- 分类器在 MSR 域的正确率 (F1=0.896) vs LLM 域 (≈0.5) 的对比图

---

---

## Baseline 实验矩阵

目标：验证"其他方法能在 LLM 生成域上选到安全代码吗？"为本文方法提供对比基准。

### Baseline 列表

| # | Baseline | 做法 | 对比维度 | 工作量 |
|---|----------|------|---------|--------|
| B0 | Random | 已完成（实验B） | 下界 | 0 |
| B1 | 分类器 P(vul) | 已完成（实验B） | 本文 post-hoc 方法 | 0 |
| B2 | Flawfinder | 对 150 条代码跑 flawfinder，选告警最少的 | 规则引擎 vs 学习型 | 低 |
| B3 | LLM 自评安全分 | 150 次 API："Rate safety 1-10"，选最高分 | LLM 内省 vs 外部分类器 | 低 |
| B4 | Safe Prompt | prompt 加安全词重新生成，选一条 | Prompt 工程 vs 分类器 | 中 |
| B5 | SVEN-style Prompt | 按 SVEN 的安全 prefix 范式构造 prompt | Prefix 引导 vs 分类器反馈 | 中 |
| B6 | Reflexion | LLM 自我反思安全审查 → 迭代修复 (NeurIPS 2023) | 自我反思 vs 外部oracle反馈 | 中 |
| B7 | CoSec-inspired | Prompt 模拟协同解码：安全规范→逐块自检 (ISSTA 2024, approx) | 逐语句安全审查 vs 迭代反馈 | 中 |

### 各 Baseline 详细说明

#### B2: Flawfinder
- 安装 `pip install flawfinder`
- 对 `all_candidates.csv` 中 150 条代码逐一运行 `flawfinder`
- 统计每条代码的告警级别和数量（按 CWE 分类）
- 选择策略：每个 prompt 的 10 条候选中，选总告警数最少的
- 评估：human safe-pick rate

#### B3: LLM 自评安全分
- 不需要重新生成代码，直接用已有的 150 条
- Prompt：`"Review this C code for security vulnerabilities. Consider buffer overflows, NULL dereference, use-after-free, integer overflow, and memory leaks. Rate its overall safety on a scale of 1-10 where 10 is perfectly secure. Output ONLY the number.\n\nCODE:\n{code}"`
- 选择策略：选自评分数最高的候选
- 评估：human safe-pick rate

#### B4: Safe Prompt
- 修改原始 prompt，注入安全约束关键词
- 模板：`"Write a SECURE C function that [原始任务]. The function MUST include: proper input validation, bounds checking, NULL checks, error handling for all memory allocations, and safe string operations. Generate only the C function code."`
- 每个 prompt 生成 10 条，选分类器 P(vul) 最低的
- 评估：human safe-pick rate + 人工标注新增的 10 条

#### B5: SVEN-style Prompt
- SVEN 原文核心：通过 prompt prefix 注入安全属性描述，引导模型生成安全代码
- 模板：`"You are an expert C programmer who writes secure, production-quality code. Always follow secure coding practices: validate all inputs, check buffer bounds, handle allocation failures, avoid undefined behavior. Generate secure C code for the following task:\n\n{原始prompt}"`
- 每个 prompt 生成 10 条，选分类器 P(vul) 最低的
- 评估：human safe-pick rate

### 预期输出对比表

| Baseline | Safe-Pick Rate | Avg P(vul) | 说明 |
|----------|---------------|------------|------|
| B0 Random | 92.9% (13/14) | — | 下界 |
| B1 分类器 | 85.7% (12/14) | — | 跨域失效 |
| B2 Flawfinder | ? | — | 规则引擎 |
| B3 LLM 自评 | ? | — | 内省能力 |
| B4 Safe Prompt | ? | — | Prompt 工程 |
| B5 SVEN-style | ? | — | Prefix 引导 |
| B6 Reflexion | ? | — | 自我反思迭代 |
| B7 CoSec-inspired | ? | — | Prompt 模拟协同解码 |
| ISR-3 (Tier 1) | ? | — | 本文方法 |

### SVEN 的引用策略

- B5 (SVEN-style Prompt) 不是真正的 SVEN——真正的 SVEN 需要 prefix-tuning 训练，B5 只是复现其 prompt 范式
- 论文中写为 "SVEN-inspired safety prefix" 或 "Prompt-based safety steering (inspired by SVEN [He & Vechev, 2023])"
- 同时引用 SVEN 原文作为 related work，说明：
  > "SVEN [He & Vechev, 2023] uses prefix-tuning to inject safety signals into the model's hidden states, requiring white-box access and training. Our approach operates through external feedback loops over black-box API calls, which is complementary and applicable to proprietary models. We compare against a prompt-based adaptation of SVEN's safety prefix as a reference point."

---

## 新方案：让安全信号真正进入生成过程

---

## Tier 1：迭代安全精炼（Iterative Safety Refinement, ISR）

**不需要训练模型，纯推理管线。目标：验证"分类器反馈能改善LLM生成"这个核心假设。**

### 核心机制

```
Prompt → LLM生成初版代码(+安全推理)
  → 分类器评分 P(vul)
  → if P(vul) > threshold:
      → 分类器 Attention 热力图 → 定位"最可疑的token span"
      → 构造反馈: "代码在[X位置]存在安全隐患，请修复：1... 2..."
      → LLM针对性修复 → 再评分
  → 循环至 P(vul) < 阈值 或 连续N次未改善 或 达到最大迭代上限
  → 保留所有迭代中最安全的版本
```

### 关键创新点

1. **Attention热力图定位**：AttentionPooling对每个token产生的注意力权重，高权重=分类器认为该token对"漏洞判断"最重要 → 映射回代码行
2. **精确定位反馈**：消融实验对比"笼统反馈"vs"attention-guided精确定位反馈"
3. **闭环学习**：安全信号不是一次性打分，而是通过反馈回路持续影响LLM的生成决策

### 实现文件

- `scripts/experiment_isr.py` — 主实验脚本
- 复用：`attention_pooling.py` (SafetyClassifier), `experiment_b.py` (API配置/score_code)

### 消融实验设计

| 配置 | 反馈方式 | 目的 |
|------|---------|------|
| ISR-0 | 无反馈，一次生成 | Baseline (=实验B) |
| ISR-1 | 笼统反馈："代码不安全，请重写" | 验证反馈本身的效果 |
| ISR-2 | Attention-guided反馈，含具体定位 | 验证精确定位的增量贡献 |
| ISR-3 | Attention-guided + 安全规范约束 | 完整系统 |

### 核心指标

- P(vul)逐轮变化曲线（应该单调下降）
- 最终P(vul) vs 基线P(vul)
- 人工标注安全率 (vs human_eval.csv)
- 迭代次数分布（多少prompt能收敛）

### 预期工作量

1-2天，API成本约300-500次调用（15 prompt × 10 candidate × 平均3轮迭代）

---

## Tier 2：分类器驱动的安全偏好对齐（Safety DPO）

**需要GPU，训练开源小模型。目标：让模型在参数层面学会生成安全代码。**

### 核心机制

```
Step 1: 偏好数据构造
  对每个C代码生成prompt → 用base model采样N条completion
  → 分类器评分 P(vul)
  → 构造偏好对: (P(vul)低 = chosen, P(vul)高 = rejected)

Step 2: DPO训练
  对 DeepSeek-Coder-1.3B / Qwen2.5-Coder-7B (LoRA) 做DPO
  → 损失函数: L = -log σ(β(log π_θ(chosen|x) - log π_ref(chosen|x)) 
                          - β(log π_θ(rejected|x) - log π_ref(rejected|x))))
  → 模型学习：对同一prompt，安全代码 > 不安全代码

Step 3: 三重评估
  - 人工标注 (human_eval.csv): 安全率
  - 分类器: P(vul) 均值
  - 静态分析: Flawfinder/CodeQL 告警数
```

### 方法创新

1. **自动化reward**：用分类器替代昂贵的人工偏好标注，使安全对齐可规模化
2. **域内对齐**：分类器训练在MSR真实漏洞数据，DPO将这种安全知识蒸馏到生成模型中
3. **闭环**：Tier 1 验证假设 → Tier 2 固化成果

### 数据需求

- 训练prompt：~300-500个C代码生成任务（可从现有MSR数据 + LLM合成）
- 每个prompt 4-8条completion → ~1200-4000个偏好对
- 验证：现有15 prompt × 10 candidate = 150条人工标注

### 模型选择

| 模型 | 参数量 | 最低显存 | 训练方式 |
|------|--------|----------|---------|
| DeepSeek-Coder-1.3B | 1.3B | 8GB | Full fine-tune |
| CodeLlama-7B | 7B + ~10M LoRA | 24GB | LoRA |
| Qwen2.5-Coder-7B | 7B + ~10M LoRA | 24GB | LoRA (推荐) |

### 预期工作量

3-5天（数据构造1天 + 训练调试2天 + 评估分析2天）

### 实现文件

- `scripts/build_preference_data.py` — 偏好对构造
- `scripts/train_dpo.py` — DPO训练（基于trl库）
- `scripts/eval_safety_dpo.py` — 三重评估

---

## 论文叙事整合

```
阶段一：对比学习预训练 (CodeBERT + InfoNCE, acc=85.7%)
阶段二：安全分类器训练 (AttentionPooling + MLP, F1=0.896)
阶段三：分类器驱动的安全生成
  ├── 实验B：后过滤baseline（验证分布偏移问题）
  ├── 实验C (ISR)：迭代安全精炼（验证反馈机制有效性）
  └── 实验D (DPO)：安全偏好对齐（将安全知识蒸馏到生成模型）
```

核心贡献陈述：
> "我们提出分类器驱动的安全反馈机制，通过注意力热力图定位缺陷并提供精确修复建议，使LLM迭代精炼代码。进一步，我们利用分类器作为自动化reward model，通过DPO将安全偏好注入开源代码模型，使模型在参数层面习得安全编码能力。"

---

## Phase 1 对比学习的真正贡献：双信号融合设计

### 问题诊断

当前 pipeline 中，对比学习（Phase 1）的 256 维投影空间几乎被架空：

| 组件 | 是否在使用 | 作用 |
|------|-----------|------|
| CodeBERT backbone | ✓ 在用 | 为分类器提供 token 表示 |
| 投影头 768→256 L2 归一化 | ✗ 未用 | 只在原型 ablation 里用了，F1=0.67 |
| 对比学习 InfoNCE 目标 | ✗ 未用 | 训练完就被丢弃 |
| 安全/漏洞原型 | ✗ 未用 | 只在 evaluate_safety.py 里对比用 |

核心问题：如果直接用原始 CodeBERT 训分类器也能达到 F1≈0.88，那 Phase 1 的对比学习贡献就只是"稍微好了点 backbone 表示"，不构成论文的核心贡献点。

### 解决方案：双信号互补架构

让对比学习嵌入空间作为**全局安全语义信号**，与分类器的**局部定位信号**形成互补：

| 信号 | 来源 | 维度 | 特点 | 角色 |
|------|------|------|------|------|
| P(vul) + Attention | 分类器（768维 token 空间） | 细粒度 | 能定位到具体 token | **缺陷定位**：告诉 LLM "哪里有问题" |
| 安全裕度 + 原型距离 | 对比嵌入（256维 CLS 空间） | 全局 | 衡量整体语义安全度 | **严重度判断**：告诉 LLM "问题有多严重" + DPO 偏好标注 |

**为什么两个信号互补**：
- 分类器过拟合 MSR 分布 → P(vul) 绝对值在 LLM 域不可靠 → 但 attention 定位相对可靠
- 对比嵌入在球面上显式拉开 safe/vul → 全局判断更鲁棒 → 但无法定位具体缺陷

### 在 ISR 反馈中的双重呈现

```
## SAFETY ANALYSIS
- Risk score: P(vul) = 0.85
- Safety embedding: distance to SAFE prototype = 0.72
                    distance to VUL prototype  = 0.31
  → This code is semantically closer to KNOWN VULNERABLE patterns

## ATTENTION HEATMAP (specific locations flagged)
[Rank 1] "while(*src) *dst++ = *src++" ████████ (0.0234)
  → Missing bounds check on destination buffer
[Rank 2] "*dst = '\0'" ██ (0.0156)  
  → No NULL validation before write

## REQUIRED FIXES
1. Add bounds checking before the copy loop
2. Validate src and dst are non-NULL
```

### 在 DPO 中的角色

DPO 需要构造偏好对 (chosen, rejected)。使用**对比嵌入裕度**而非 P(vul) 作为偏好信号：

- `margin = sim(emb, safe_proto) - sim(emb, vul_proto)`
- chosen = margin 最高的 completion
- rejected = margin 最低的 completion

理由：对比嵌入裕度在 256 维归一化空间中衡量，对 LLM 域和 MSR 域的表示差异有一定鲁棒性（因为是全局语义距离，不受具体 token 分布偏移影响）。

### 论文叙事更新

> "Phase 1 对比学习不仅提升了 backbone 的 token 表示质量（pair matching 85.7%），更重要的是建立了一个结构化的安全语义嵌入空间——safe/vul 代码在超球面上被显式分离。该空间在 Phase 3 中作为全局安全奖励信号，与分类器的细粒度注意力定位形成互补：对比嵌入判断'是否需要修复'，分类器注意力定位'需要修复哪里'。两者共同构成了安全引导生成的双信号系统。"

### 需要补的消融实验

| 消融 | 操作 | 目的 |
|------|------|------|
| 原始 CodeBERT vs 对比学习 CodeBERT | 用原始 CodeBERT 重新训分类器，对比 F1 | 量化对比学习对 backbone 的提升 |
| 仅 P(vul) vs 仅 Margin vs 双信号 | ISR 反馈中分别使用单一信号和双信号 | 验证双信号互补的必要性 |

---

## 下一步

1. [x] 实现 Tier 1: `experiment_isr.py`
2. [x] 实现 Tier 1 分析: `analyze_isr.py`
3. [x] 对比学习双信号融合设计文档化
4. [x] Baseline 矩阵设计（B0-B5）
5. [ ] 跑 B2: Flawfinder 静态分析
6. [ ] 跑 B3: LLM 自评安全分
7. [ ] 跑 B4: Safe Prompt 生成
8. [ ] 跑 B5: SVEN-style Prompt 生成
9. [ ] 将对比嵌入裕度集成到 ISR 反馈中
10. [ ] 补消融：原始 CodeBERT vs 对比学习 CodeBERT 分类器 F1 对比
11. [ ] 在服务器运行 ISR 实验
12. [ ] 分析结果，确认反馈机制有效性
13. [ ] 基于 Tier 1 结论决定 Tier 2 启动时机
14. [ ] 更新论文文档

### 当前状态（2026-05-26）

**Tier 1 已暂停**。导师要求先跑其他方法作为 baseline 对比，完成后再启动 ISR 实验。

Tier 1 及格线（回来后对照）：
1. P(vul) 随迭代单调下降，最终值 < ISR-0 baseline
2. ISR-2 (精确反馈) > ISR-1 (笼统反馈)，证明 attention 定位有效
3. 人工标注安全率 > 实验B 的 85.7%
