# ISR 泛化性扩展分析：从 C 内存安全到 Python SQL 注入

## 审稿人潜在质疑

当前 ISR 只在 C 语言内存安全漏洞上验证。内存安全漏洞（缓冲区溢出、UAF 等）有明确的静态特征——`memcpy`、`strcpy`、`malloc`、`free` 等函数名本身就是强信号。审稿人可能质疑：

> "ISR 的分类器注意力定位是否只对'函数名即漏洞信号'的场景有效？换成逻辑漏洞或注入攻击，分类器能提供有用的注意力信号吗？"

## 建议方案：小规模跨领域 Case Study

### 目标

证明 ISR 框架（外部分类器 + 注意力引导反馈 + 迭代修复）是**范式级贡献**，可迁移到不同语言和漏洞类型，而非 C 内存安全的特例。

### 范围

不做大规模实验，仅仅 5-10 个 prompt 的概念验证（Proof-of-Concept）。

### 漏洞类型选择：Python SQL 注入

| 维度 | Python SQL 注入 vs C 内存安全 |
|------|------------------------------|
| 语言 | 动态 vs 静态 / 无指针 vs 有指针 |
| 漏洞信号 | 字符串拼接模式 (`f"...%s"`) vs 函数名 (`memcpy`) |
| 安全模式 | 参数化查询 vs 边界检查 |
| 分类器挑战 | 需要识别数据流模式而非关键字 |

如果 ISR 能在这两个差异极大的领域都起作用，泛化性论证就很强。

### 实验设计

**prompt 设计**（5 个，覆盖典型注入场景）：
1. `search_user_by_name` — 根据用户名查数据库（最经典的注入）
2. `insert_order_record` — 向订单表插入数据（多字段拼接）
3. `filter_products_by_category` — 按类别筛选商品（WHERE 子句拼接）
4. `update_user_password` — 更新用户密码（SET 子句拼接）
5. `login_by_email` — 邮箱登录查询（认证查询，高危）

**分类器方案**（三选一，按难度排序）：

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| A. 规则引擎 | `sqlparse` + 启发式规则（检测字符串拼接 + execute()） | 最快，零训练成本 | 无注意力权重，只能做 ISR-0/ISR-1 |
| B. CodeBERT fine-tune | 在 Python SQL 注入数据集上微调 CodeBERT + AttentionPooling | 有注意力权重，能做 ISR-2/ISR-3 | 需要标注数据（~500 条即可） |
| C. 现成 LLM 当分类器 | 用 Claude/GPT 做 zero-shot 安全评分 + 逐行标注 | 最灵活，无需训练 | 又回到 LLM 自评的循环了 |

**建议选 A 起步**：ISR-0 vs ISR-1 的对照本身就验证核心假设——外部信号即使模糊（规则级别）也比没有强。如果需要 ISR-2/ISR-3 的注意力实验再加 B。

**对照**：
| 配置 | 反馈 | 对等 C 内存安全配置 | 验证问题 |
|------|------|---------------------|----------|
| ISR-0 | 无 | ISR-0 | Python 下界 |
| ISR-1 | 规则引擎通用告警 | ISR-1 | 外部信号在另一领域是否同样触发改进 |
| ISR-2 | 规则引擎 + 定位具体行 | ISR-2 | 精准定位是否跨领域有效（需方案 B） |

**评估**：
- 分类器评分趋势对比（ISR-0 → ISR-1 → ISR-2 是否单调下降）
- 快速人工扫视（不需要 320 条盲化，作者自审即可）
- 案例展示（put 1-2 个 before/after 代码对比在论文里）

### 论文位置

在 §5 Discussion 末尾新增一小段（约半页）：

> **§5.X Case Study: Beyond C Memory Safety.** To assess generalizability, we replicate ISR on 5 Python SQL injection tasks using a rule-based safety classifier. ISR-1 improves the safe-generation rate from X% to Y%, confirming that external feedback remains effective across languages and vulnerability types. While the rule-based classifier lacks fine-grained attention, the iteration activation effect—the dominant component of ISR's gain—replicates consistently. This suggests ISR's decoupling paradigm generalizes to any vulnerability domain where an independent safety oracle exists, even a coarse one.

### 工作量估算

| 步骤 | 时间 |
|------|------|
| 写 5 个 Python prompt | 20 分钟 |
| 搭规则引擎分类器（sqlparse + 正则） | 1 小时 |
| 跑 ISR-0 + ISR-1（各 5 prompt × 5 candidates） | 1 小时 |
| 人工扫视 + 统计 | 30 分钟 |
| 写入论文 | 30 分钟 |
| **总计** | **~3.5 小时** |

### 风险

1. **规则引擎太粗糙**：如果检测准确率极低（50% 随机级别），ISR-1 可能不产生改进 → 反而说明需要更好的分类器，这是 honest finding，不是失败
2. **Python 代码生成质量差异**：不同 LLM 对 Python 的理解水平不同，可能需要固定模型
3. **审稿人可能要求更多**：5 个 prompt 可能被认为太少 → 提前在论文中定位为 "illustrative case study" 而非 "comprehensive evaluation"

### 不做这个的风险

审稿人可能直接给 "limited generalizability" 的弱点，要求补实验（major revision），那时补比现在补被动得多。

### 结论

**性价比极高**：3.5 小时的投入，消除一个审稿人的主要攻击面，同时提升论文从 "method for a specific problem" 到 "generalizable framework" 的定位。
