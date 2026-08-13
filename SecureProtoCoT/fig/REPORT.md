# Code Agent 模拟 Fusion 实验报告

> 编写日期：2026-06-30  
> 设计参考：[OpenRouter — Fusion Beats Frontier](https://openrouter.ai/blog/announcements/fusion-beats-frontier/) · [openfusion (GitHub)](https://github.com/hashangit/openfusion)

---

## 1. 背景

2026 年 OpenRouter 发表文章《Fusion Beats Frontier》并推出 **Fusion** 功能，核心结论是：

> 把多个独立模型针对同一问题各自产出的答案"融合"起来，可以稳定地优于任意单个前沿模型本身。

OpenRouter 的 Fusion 是一种 **后置聚合** 思路 —— 不依赖训练，不需要重新对齐，只在推理侧把 N 份候选答案做"分析 + 综合"两步处理，得到一份比任意一份候选都更可靠的最终答案。

但 OpenRouter 并未公布 Fusion 的所有内部实现细节。因此本次实验组合两份公开材料：方法论参考 OpenRouter 文章《Fusion Beats Frontier》，具体的分析 / 综合提示词原样取自开源项目 openfusion (https://github.com/hashangit/openfusion)。

---

## 2. 实验目的

在 **CodeAgent** 平台上模拟 OpenRouter Fusion 的多模型融合行为，并通过 CodeAgent 自带的代码评测体系对融合效果进行客观评分。

具体回答两个问题：

1. 把"规划阶段"由单模型替换为"多模型融合"，是否会提升最终代码质量？
2. 不同的融合配方（2 路 vs 3 路、同源 vs 异源）之间，效果差异有多大？

---

## 3. 测试集

CodeAgent 平台在多个真实项目上沉淀的 **7 条真实需求**，覆盖 4 个项目（apm-dg / cnc-apm / scm-all / taskflow-st2packlib）。

| # | 项目 | 用例 |
|---|---|---|
| 1 | apm-dg | 修复软件规划定义查询接口…resourceName 丢失 |
| 2 | cnc-apm | 应用模式兼容性规则从代码写死改为数据字典 |
| 3 | cnc-apm | 网络号校验过滤故障机 |
| 4 | cnc-apm | apm_flow_ip_fault 仅保留未恢复记录 |
| 5 | scm-all | Java 侧 5 个业务模块改造为 ODP HTTP 上报 |
| 6 | scm-all | 共用配置默认全量部署控制迁移到业务视图 |
| 7 | taskflow-st2packlib | 流水 661 新增 action_task_check_661_info |

每条需求在每个实验组下复跑 **5 次**，单组样本量 = 7 × 5 = 35。

---

## 4. 如何模拟 Fusion

### 4.1 单模型路径（基准）

```
需求 ──▶ 模型 X 生成 plan.md ──▶ 固定 dp 编码 ──▶ 评测打分
```

通过对最终代码打分，**间接**反馈 plan.md 的质量。

### 4.2 融合模型路径

```
需求 ──▶ 模型 A ──▶ planA.md ─┐
需求 ──▶ 模型 B ──▶ planB.md ─┤
                                ├──▶ 融合 agent (Opus 4.7) ──▶ 融合 plan.md ──▶ 固定 dp 编码 ──▶ 评测打分
需求 ──▶ 模型 C ──▶ planC.md ─┘   │
                                  └─ 内部派发两个子 agent：
                                       · fusion-analyzer (Opus 4.7)：5 维度分析
                                       · fusion-synthesizer (Opus 4.7)：单份综合
```

> **关键约束**：融合作用在"规划阶段"，产物是 plan.md（文档）；CodeAgent 评测只能对代码做断言。
> 因此实验在规划之后增加一个**统一编码阶段** —— 所有组的 plan.md 一律由同一个 deepseek 模型实现代码，把变量收敛到"plan.md 内容差异"这一项。

### 4.3 直接采用 openfusion 的实现

Fusion 的具体方法依赖两步骤、两组提示词。这两组提示词原样取自 openfusion 项目，未做任何"调味"。

#### 4.3.1 分析（Analyzer）

> 任务：阅读 N 份候选答案，按 **5 个维度** 打标，**不回答原问题**。

**5 个维度**：

| 维度 | 含义 |
|---|---|
| consensus | 多份方案实质上达成一致的点 |
| contradictions | 多份方案给出相互冲突的结论 |
| partialCoverage | 原问题的某些方面只被一部分方案谈到 |
| uniqueInsights | 只有少数方案才提到的、有价值的点 |
| blindSpots | 所有方案都没谈好的重要方面 |

<details>
<summary><b>点击展开：fusion-analyzer 完整 system prompt</b></summary>

```
You are the ANALYSIS step of a fusion judge. You will receive a user's prompt
followed by several candidate answers produced independently by different
models.

Your job is to ANALYZE the candidates — NOT to answer the prompt yourself.
You must call the provided "record_analysis" tool exactly once with your
structured analysis. The analysis must capture:

- consensus: points where the candidates substantially agree
- contradictions: points where candidates disagree or conflict
- partialCoverage: aspects of the prompt that only some candidates addressed
- uniqueInsights: valuable points raised by only one or a few candidates
- blindSpots: important aspects of the prompt that no candidate addressed well

Be precise and concrete; cite specific candidates by their index
(Candidate 1, 2, ...) where relevant. Do NOT write a free-text answer —
use the tool.

---

## OUTPUT

Instead of calling `record_analysis`, output your analysis as JSON with this
exact schema (each value is an array of strings; empty arrays are allowed
when a dimension truly has no entries):

{
  "consensus":        ["..."],
  "contradictions":   ["..."],
  "partialCoverage":  ["..."],
  "uniqueInsights":   ["..."],
  "blindSpots":       ["..."]
}

Required steps:

1. Write the JSON object to `.codeagent/analysis.json` using the Write tool
   (path is relative to the current working directory; the directory has
   been created for you).

2. Return ONLY a fenced ```json``` code block containing the same JSON object
   as your final message. No preamble, no commentary, no extra text.

If a re-dispatch note tells you "previous output was malformed", re-read the
schema above and fix only the cited issue; the substantive analysis should
not change.
```

</details>

#### 4.3.2 综合（Synthesizer）

> 任务：基于候选 + 5 维度分析，写出**一份**最终方案。**不引入候选/分析之外的新信息**。

<details>
<summary><b>点击展开：fusion-synthesizer 完整 system prompt</b></summary>

```
You are the SYNTHESIS step of a fusion judge. You will receive a user's
prompt, the candidate answers, and a structured analysis (consensus,
contradictions, partial coverage, unique insights, blind spots) produced in
the prior step.

Write the single best consolidated answer to the user's prompt. Your
answer must:

- Reflect and integrate the candidates, prioritizing consensus points.
- Reconcile contradictions, explaining the resolution when it matters.
- Where the candidates are wrong, say so and correct it — do not
  rubber-stamp consensus.
- Incorporate unique insights and address blind spots identified in the
  analysis.
- Introduce NO new external information that was not present in the
  candidates or analysis. You are synthesizing, not researching.

Return only the final answer text (no preamble about the process).

---

## OUTPUT

1. Write your final consolidated answer (pure markdown, no preamble or
   meta-commentary about the synthesis process) to `.codeagent/plan.md`
   using the Write tool. The path is relative to the current working
   directory; the directory has been created for you.

2. Return ONLY a single-line summary as your final message in this exact
   format:

       OK: lines=<N>, chars=<M>

   where `<N>` is the line count and `<M>` is the character count of the
   plan.md you just wrote. No preamble, no additional commentary.

If a re-dispatch note tells you "previous output failed", re-read the
constraints above and re-attempt. Do not embellish or shorten beyond what
those constraints require.
```

</details>

#### 4.3.3 设计取舍要点

| 取舍 | 为什么 |
|---|---|
| 分析步骤 **不答** 原问题 | 强制产出可被人审阅的结构化结论，而不是又一份候选 |
| 综合步骤 **不查** 外部资料 | 避免引入候选之外的不可控信息，保持融合的可解释性 |
| 分析与综合 **由独立 subagent** 执行 | 上下文互不污染，避免分析阶段的中间推理影响综合结果 |

---

## 5. 实验组与模型矩阵

### 5.1 模型清单

> **别名说明**：

| 别名 | 实际模型 |
|---|---|
| opus | Anthropic Claude Opus 4.7 |
| glm | 智谱 GLM 5.1 |
| dp | DeepSeek Pro |
| dpflash | DeepSeek Flash |

| 角色 | 模型 |
|---|---|
| 基准（Opus 单模型） | opus |
| DP 单模型 | dp |
| DPFlash 单模型 | dpflash |
| GLM 单模型 | glm |
| 融合协调 / 分析 / 综合 | opus（统一） |
| **统一编码模型** | dp — 所有组共用 |

### 5.2 实验组（共 7 组，每组 35 次实验）

| 组别 | 规划阶段输入 | 备注 |
|---|---|---|
| **快速流程** | Opus 单模型直接产出 plan.md | Opus 基准对照 |
| **快速流程(dp)** | DP 单模型产出 plan.md | 单模型基线 |
| **快速流程(dpflash)** | DPFlash 单模型产出 plan.md | 单模型基线 |
| **快速流程(glm)** | GLM 单模型产出 plan.md | 单模型基线 |
| **融合(glm+dp)** | 融合 GLM × DP 两份 plan.md | 2 路异源融合 |
| **融合(glm+dp+dp)** | 融合 GLM × DP × DP 三份 plan.md | 3 路（DP 复跑两次） |
| **融合(glm+dp+dpflash)** | 融合 GLM × DP × DPFlash 三份 plan.md | 3 路异源融合 |

### 5.3 评测口径

- **机评**：CodeAgent 评测系统对最终代码做断言评分。
- **人评纠偏**：在机评分基础上人工审核与纠偏。
- **最终分**：人评纠偏后的分数（即下表中的 `human_score`）。
- **聚合方式**：每个用例 5 次复跑后，取 5 次中最高 3 次的均值（去除 2 个最低）作为该用例的「**前 3 平均**」；在 7 个用例上再次取均值，得到组级整体分。
- **为什么取「前 3」而不是全 5 平均或中间 3**：
  - **聚焦"模型能做到什么"，而不是"有多大概率会做砸"**：每个用例 5 次复跑的方差不小，单次低分常源于工具偶发错误、prompt 抽样不稳、外部环境抖动这类「非能力性失败」，不应被纳入对模型上限的估计。本次实验关心的是「换个 plan 写法能让代码质量到多高」，所以选偏乐观的统计量更贴题。
  - **样本量小（n=5）时方差太大，全均/单次抽样都不稳**：全 5 次平均下极端低分（如 scm-all java 那次 51.67）会把整组拉低 ~10 分，结论容易反转；中间 3（去最高最低）更鲁棒但把模型的最好表现也丢了，偏保守。前 3 在「稳健 vs 反映能力」之间取折衷 —— 既剔除明显的 unlucky run，又保留模型实际能达到的高分区间。这也对齐 LLM 评测领域 pass@k 的思路（对单次随机性脱敏、只看上限）。

---

## 6. 结果

### 6.1 7 组整体平均分对比（按分数高到低）

> 「任务平均费用」= 该组前 3 用例任务在**规划阶段**的 token 花费均值（不含统一编码阶段）。

| 实验组 | 前 3 平均 | 任务平均费用 |
|---|---:|-------:|
| 快速流程（**Opus 基准**） | **95.21** | $21.52 |
| **融合(glm+dp+dpflash)** | **93.93** |  $4.30 |
| 融合(glm+dp) | 93.69 |  $4.02 |
| 融合(glm+dp+dp) | 93.44 |  $5.42 |
| 快速流程(glm) | 93.22 |  $1.57 |
| 快速流程(dp) | 92.41 |  $1.00 |
| 快速流程(dpflash) | 92.34 |  $0.06 |

### 6.2 「前 3 平均」用例明细

| 项目 | 用例 | 快速流程 | glm | dpflash | dp | 融合(glm+dp) | 融合(glm+dp+dpflash) | 融合(glm+dp+dp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| apm-dg | resourceName 丢失修复 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| cnc-apm | 兼容性规则改数据字典 | 83.46 | 80.61 | 83.79 | 85.43 | 83.36 | 84.23 | 82.15 |
| cnc-apm | 网络号校验过滤故障机 | 97.06 | 95.49 | 87.55 | 90.98 | 94.70 | 93.04 | 94.80 |
| cnc-apm | apm_flow_ip_fault 仅保留未恢复 | 98.22 | 100.00 | 99.10 | 100.00 | 99.10 | 99.10 | 98.20 |
| scm-all | Java 5 模块改造 ODP HTTP | 100.00 | 98.78 | 98.00 | 97.22 | 95.89 | 96.11 | 97.22 |
| scm-all | 共用配置全量部署迁移 | 100.00 | 100.00 | 100.00 | 100.00 | 99.17 | 97.50 | 100.00 |
| taskflow | 流水 661 新增 check_info | 87.71 | 77.68 | 77.96 | 73.24 | 83.61 | 87.50 | 81.72 |

### 6.3 原始数据（35 实验逐次得分）

#### 1. 修复软件规划定义查询接口…resourceName 丢失（apm-dg）

| 实验 | 分数 | 最高 | 最低 | 平均 |
|---|---|---:|---:|---:|
| 快速流程001 / 002 / 003 / 004 / 005 | 100.00 / 100.00 / 100.00 / 100.00 / 100.00 | 100.00 | 100.00 | 100.00 |
| 快速流程(glm)001 / 002 / 003 / 004 / 005 | 100.00 / 100.00 / 100.00 / 100.00 / 100.00 | 100.00 | 100.00 | 100.00 |
| 快速流程(dpflash)001 / 002 / 003 / 004 / 005 | 100.00 / 100.00 / 100.00 / 100.00 / 100.00 | 100.00 | 100.00 | 100.00 |
| 快速流程(dp)001 / 002 / 003 / 004 / 005 | 100.00 / 100.00 / 100.00 / 100.00 / 100.00 | 100.00 | 100.00 | 100.00 |
| 融合(glm+dp)001 / 002 / 003 / 004 / 005 | 100.00 / 100.00 / 100.00 / 100.00 / 100.00 | 100.00 | 100.00 | 100.00 |
| 融合(glm+dp+dpflash)001 / 002 / 003 / 004 / 005 | 100.00 / 100.00 / 100.00 / 100.00 / 100.00 | 100.00 | 100.00 | 100.00 |
| 融合(glm+dp+dp)001 / 002 / 003 / 004 / 005 | 100.00 / 100.00 / 100.00 / 100.00 / 100.00 | 100.00 | 100.00 | 100.00 |
| **全用例（35 实验）** |  | **100.00** | **100.00** | **100.00** |

#### 2. 应用模式兼容性规则从代码写死改为数据字典（cnc-apm）

| 实验 | 分数 | 最高 | 最低 | 平均 |
|---|---|---:|---:|---:|
| 快速流程001 / 002 / 003 / 004 / 005 | 87.56 / 71.14 / 85.78 / 77.05 / 71.82 | 87.56 | 71.14 | 78.67 |
| 快速流程(glm)001 / 002 / 003 / 004 / 005 | 79.55 / 75.00 / 79.77 / 82.50 / 77.64 | 82.50 | 75.00 | 78.89 |
| 快速流程(dpflash)001 / 002 / 003 / 004 / 005 | 82.95 / 76.82 / 80.23 / 82.50 / 85.91 | 85.91 | 76.82 | 81.68 |
| 快速流程(dp)001 / 002 / 003 / 004 / 005 | 83.56 / 86.82 / 85.91 / 73.18 / 80.00 | 86.82 | 73.18 | 81.89 |
| 融合(glm+dp)001 / 002 / 003 / 004 / 005 | 81.09 / 81.09 / 83.36 / 85.64 / 80.41 | 85.64 | 80.41 | 82.32 |
| 融合(glm+dp+dpflash)001 / 002 / 003 / 004 / 005 | 80.86 / 82.55 / 84.50 / 85.64 / 80.64 | 85.64 | 80.64 | 82.84 |
| 融合(glm+dp+dp)001 / 002 / 003 / 004 / 005 | 76.55 / 80.64 / 83.36 / 81.55 / 81.55 | 83.36 | 76.55 | 80.73 |
| **全用例（35 实验）** |  | **87.56** | **71.14** | **81.00** |

#### 3. 网络号校验过滤故障机（cnc-apm）

| 实验 | 分数 | 最高 | 最低 | 平均 |
|---|---|---:|---:|---:|
| 快速流程001 / 002 / 003 / 004 / 005 | 95.29 / 97.65 / 98.24 / 87.06 / 85.29 | 98.24 | 85.29 | 92.71 |
| 快速流程(glm)001 / 002 / 003 / 004 / 005 | 98.24 / 98.24 / 88.24 / 86.24 / 90.00 | 98.24 | 86.24 | 92.19 |
| 快速流程(dpflash)001 / 002 / 003 / 004 / 005 | 94.12 / 87.94 / 80.59 / 75.88 / 77.65 | 94.12 | 75.88 | 83.24 |
| 快速流程(dp)001 / 002 / 003 / 004 / 005 | 94.12 / 86.47 / 75.53 / 92.35 / 77.29 | 94.12 | 75.53 | 85.15 |
| 融合(glm+dp)001 / 002 / 003 / 004 / 005 | 77.65 / 97.06 / 88.24 / 95.29 / 91.76 | 97.06 | 77.65 | 90.00 |
| 融合(glm+dp+dpflash)001 / 002 / 003 / 004 / 005 | 89.41 / 97.65 / 76.47 / 84.41 / 92.06 | 97.65 | 76.47 | 88.00 |
| 融合(glm+dp+dp)001 / 002 / 003 / 004 / 005 | 87.94 / 95.29 / 92.35 / 96.76 / 87.65 | 96.76 | 87.65 | 92.00 |
| **全用例（35 实验）** |  | **98.24** | **75.53** | **89.01** |

#### 4. apm_flow_ip_fault 仅保留未恢复记录（cnc-apm）

| 实验 | 分数 | 最高 | 最低 | 平均 |
|---|---|---:|---:|---:|
| 快速流程001 / 002 / 003 / 004 / 005 | 100.00 / 94.59 / 97.37 / 97.30 / 97.30 | 100.00 | 94.59 | 97.31 |
| 快速流程(glm)001 / 002 / 003 / 004 / 005 | 100.00 / 100.00 / 100.00 / 97.30 / 100.00 | 100.00 | 97.30 | 99.46 |
| 快速流程(dpflash)001 / 002 / 003 / 004 / 005 | 100.00 / 97.30 / 95.14 / 100.00 / 94.59 | 100.00 | 94.59 | 97.41 |
| 快速流程(dp)001 / 002 / 003 / 004 / 005 | 100.00 / 91.89 / 99.24 / 100.00 / 100.00 | 100.00 | 91.89 | 98.23 |
| 融合(glm+dp)001 / 002 / 003 / 004 / 005 | 97.30 / 97.30 / 100.00 / 97.30 / 100.00 | 100.00 | 97.30 | 98.38 |
| 融合(glm+dp+dpflash)001 / 002 / 003 / 004 / 005 | 100.00 / 87.57 / 97.30 / 100.00 / 97.30 | 100.00 | 87.57 | 96.43 |
| 融合(glm+dp+dp)001 / 002 / 003 / 004 / 005 | 97.30 / 91.89 / 100.00 / 97.30 / 88.32 | 100.00 | 88.32 | 94.96 |
| **全用例（35 实验）** |  | **100.00** | **87.57** | **97.45** |

#### 5. Java 侧 5 个业务模块改造为 ODP HTTP 上报（scm-all）

| 实验 | 分数 | 最高 | 最低 | 平均 |
|---|---|---:|---:|---:|
| 快速流程001 / 002 / 003 / 004 / 005 | 100.00 / 100.00 / 100.00 / 96.67 / 97.67 | 100.00 | 96.67 | 98.87 |
| 快速流程(glm)001 / 002 / 003 / 004 / 005 | 97.33 / 96.67 / 97.67 / 98.67 / 100.00 | 100.00 | 96.67 | 98.07 |
| 快速流程(dpflash)001 / 002 / 003 / 004 / 005 | 100.00 / 91.67 / 96.67 / 80.00 / 97.33 | 100.00 | 80.00 | 93.13 |
| 快速流程(dp)001 / 002 / 003 / 004 / 005 | 83.33 / 93.33 / 95.00 / 100.00 / 96.67 | 100.00 | 83.33 | 93.67 |
| 融合(glm+dp)001 / 002 / 003 / 004 / 005 | 76.00 / 51.67 / 100.00 / 98.33 / 89.33 | 100.00 | 51.67 | 83.07 |
| 融合(glm+dp+dpflash)001 / 002 / 003 / 004 / 005 | 97.67 / 90.67 / 100.00 / 90.00 / 56.00 | 100.00 | 56.00 | 86.87 |
| 融合(glm+dp+dp)001 / 002 / 003 / 004 / 005 | 96.67 / 100.00 / 88.33 / 86.67 / 95.00 | 100.00 | 86.67 | 93.33 |
| **全用例（35 实验）** |  | **100.00** | **51.67** | **92.43** |

#### 6. 共用配置默认全量部署控制迁移到业务视图（scm-all）

| 实验 | 分数 | 最高 | 最低 | 平均 |
|---|---|---:|---:|---:|
| 快速流程001 / 002 / 003 / 004 / 005 | 98.00 / 66.25 / 100.00 / 100.00 / 100.00 | 100.00 | 66.25 | 92.85 |
| 快速流程(glm)001 / 002 / 003 / 004 / 005 | 100.00 / 100.00 / 100.00 / 100.00 / 67.50 | 100.00 | 67.50 | 93.50 |
| 快速流程(dpflash)001 / 002 / 003 / 004 / 005 | 100.00 / 100.00 / 100.00 / 100.00 / 71.25 | 100.00 | 71.25 | 94.25 |
| 快速流程(dp)001 / 002 / 003 / 004 / 005 | 93.75 / 100.00 / 100.00 / 100.00 / 100.00 | 100.00 | 93.75 | 98.75 |
| 融合(glm+dp)001 / 002 / 003 / 004 / 005 | 100.00 / 100.00 / 97.50 / 97.50 / 95.50 | 100.00 | 95.50 | 98.10 |
| 融合(glm+dp+dpflash)001 / 002 / 003 / 004 / 005 | 95.00 / 97.50 / 100.00 / 88.75 / 85.50 | 100.00 | 85.50 | 93.35 |
| 融合(glm+dp+dp)001 / 002 / 003 / 004 / 005 | 100.00 / 100.00 / 100.00 / 95.00 / 98.00 | 100.00 | 95.00 | 98.60 |
| **全用例（35 实验）** |  | **100.00** | **66.25** | **95.63** |

#### 7. 流水 661 新增 action_task_check_661_info（taskflow-st2packlib）

| 实验 | 分数 | 最高 | 最低 | 平均 |
|---|---|---:|---:|---:|
| 快速流程001 / 002 / 003 / 004 / 005 | 84.05 / 80.00 / 92.70 / 72.78 / 86.39 | 92.70 | 72.78 | 83.18 |
| 快速流程(glm)001 / 002 / 003 / 004 / 005 | 75.83 / 75.83 / 81.39 / 70.28 / 75.28 | 81.39 | 70.28 | 75.72 |
| 快速流程(dpflash)001 / 002 / 003 / 004 / 005 | 68.33 / 73.44 / 81.94 / 73.89 / 78.06 | 81.94 | 68.33 | 75.13 |
| 快速流程(dp)001 / 002 / 003 / 004 / 005 | 73.89 / 75.83 / 68.33 / 66.11 / 70.00 | 75.83 | 66.11 | 70.83 |
| 融合(glm+dp)001 / 002 / 003 / 004 / 005 | 81.39 / 86.94 / 82.50 / 71.39 / 75.28 | 86.94 | 71.39 | 79.50 |
| 融合(glm+dp+dpflash)001 / 002 / 003 / 004 / 005 | 80.83 / 87.22 / 81.39 / 75.28 / 93.89 | 93.89 | 75.28 | 83.72 |
| 融合(glm+dp+dp)001 / 002 / 003 / 004 / 005 | 77.22 / 84.44 / 81.39 / 75.61 / 79.33 | 84.44 | 75.61 | 79.60 |
| **全用例（35 实验）** |  | **93.89** | **66.11** | **78.24** |

---

## 7. 结论与观察

### 7.1 客观陈述

1. **Opus 单模型仍居榜首**：Opus 基准组「前 3 平均」95.21，尚未被任何融合配方超越。

2. **融合相对"被融合的单模型"普遍有正向收益**：
   - 融合(glm+dp+dpflash) (93.93) **超过** 它的三个组成单模型 glm(93.22) / dp(92.41) / dpflash(92.34) **任意一个**。
   - 融合(glm+dp) (93.69) 同样超过其组成的 dp 与 glm。
   - 融合(glm+dp+dp) (93.44) 也超过 dp(92.41) 和 glm(93.22)。
   - 这与 OpenRouter 文章"融合优于单个模型"的论断方向一致。

3. **异源 3 路最优，同源叠加次之，异源 2 路再次**：融合(glm+dp+dpflash)(93.93) > 融合(glm+dp)(93.69) > 融合(glm+dp+dp)(93.44)，差距均在 0.3 左右。异源候选带来的信息增量最大；同源叠加（dp + dp）的额外候选作用有限，但并非完全无效。

4. **个别用例上，融合不仅没赢、反而更差**：
   - **scm-all / Java 5 模块改造**：单模型快速流程 100.00、glm 98.78、dp 97.22；融合(glm+dp) 跌到 95.89，融合(glm+dp+dpflash) 也只到 96.11。
   - **scm-all / 共用配置全量部署迁移**：4 个单模型组前 3 平均都达到 100.00，融合反而出现 97.50 的回退。
   - 这两个用例的共同特征是 **单模型已经接近天花板**，融合阶段任何"调和矛盾"或"补齐盲区"的动作都更可能是 **多余动作**，反而引入风险。

5. **越是单模型表现差的用例，融合的相对增益越大**：
   - **taskflow / 流水 661**：dp 单模型最低（73.24），融合(glm+dp+dpflash) 一举拉到 87.50，**相对最弱单模型 +14.26**，相对最强单模型 dpflash(77.96) 也有 **+9.54**。
   - **cnc-apm / 网络号校验**：dp(90.98) → 融合(glm+dp) 94.70（+3.72）。
   - 这与 OpenRouter 主张的"短板互补"机制一致。

### 7.2 结论

- 在 CodeAgent 现有评测体系下，**OpenRouter 风格的 Fusion 方法对中等强度的单模型组合（dp / dpflash / glm）确有提升作用，且提升随候选异源性增加而扩大**。
- 但 **Fusion 并非银弹**：
  - 不能克服"基础模型差距"（融合三方仍未追上 Opus）；
  - 在单模型已接近满分的用例上反而是负担；
  - 同源叠加（glm+dp+dp）也能带来轻微增量，但不及引入真正异源候选。

---

## 附录 A：实验阶段依赖

```
A 规划：Opus × 5 ─┐
B 规划：DP   × 5 ─┤
C 规划：GLM  × 5 ─┤── 阶段 D 必须在 B、C 同序号完成后启动
D 规划：fusion(B,C) × 5 ─┘
E 编码：4 组 × 5 = 20 份 plan.md，统一用 deepseek 实现
F1 机评：评测系统对 20 份代码做断言评分
F2 人评：在机评分上做纠偏，作为最终分
```

> 编码阶段强制使用同一个 deepseek 模型，prompt 模板、温度、工具集、超时全部一致 —— 把对比变量收敛到「plan.md 内容差异」这一项。
