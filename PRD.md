# PRD：Probabilistic G-BERT: System Architecture V4 (Entity-Aware & One-Pass Edition)

## Abstract

现有的文本嵌入模型（如 BERT, E5）通常将语义映射为欧氏空间中的**确定性点向量**。这种**各向同性（Isotropic）**的假设忽略了人类记忆与情感的一个核心特征：**"强度即确定性" (Intensity implies Certainty)**。

为此，我们提出 **Probabilistic G-BERT V4**。不同于学习静态向量，我们将每个文本建模为超球面上的 **Von Mises-Fisher (vMF)** 概率分布。

1. **理论基础：** 从欧氏空间的点假设转向超球面上的**概率分布**假设。
2. **数据策略：** 引入 **Soft Label Max-Norm**，将"强度"定义为标签分布的几何尖锐程度。使用 **Character Offsets** 解决 Token 对齐问题。
3. **架构创新：** 设计了 **Token-Level Bottlenecked Tri-Branch** 网络，采用 **Project-then-Pool** 策略。
4. **训练机制：** 提出了 **Mass-Adaptive Temperature Scaling (MATS)**，从 vMF 最大似然估计推导动态对比学习温度。
5. **V4 核心变更：** **Supervised Multi-Granularity Learning** — 通过 **Sample Flattening** 和 **Character-Offset Entity Masks** 实现实体级监督训练。

---

## 1. Theoretical Formulation (理论形式化)

### 1.1 从点向量到概率分布

传统嵌入模型假设文本 $x$ 的表征是一个确定性向量：
$$\mathbf{z} = f_{\theta}(x) \in \mathbb{R}^d$$

我们**放弃这一假设**，转而假设文本的潜在表征服从 **Von Mises-Fisher (vMF) 分布**：

$$p(\mathbf{z}|x) = C_d(\kappa) \cdot \exp(\kappa \cdot \boldsymbol{\mu}^\top \mathbf{z})$$

其中：

| 参数 | 符号 | 物理意义 | 统计意义 |
|------|------|----------|----------|
| 均值方向 | $\boldsymbol{\mu}$ | 语义核心 | 分布的中心方向 ($\|\boldsymbol{\mu}\| = 1$) |
| 专注度 | $\kappa$ | 物理质量 | 分布的尖锐程度（确定性） |
| 隐变量 | $\mathbf{z}$ | 表征向量 | 从分布中采样的样本 |

$$C_d(\kappa) = \frac{\kappa^{d/2-1}}{(2\pi)^{d/2} I_{d/2-1}(\kappa)}$$

为归一化常数，$I_v(\cdot)$ 为修正贝塞尔函数。

### 1.2 物理映射：质量与专注度的等价性

我们将认知心理学中的"记忆强度"映射为统计学中的"专注度"：

$$\text{Intensity}(x) \;\longleftrightarrow\; \kappa(x)$$

**核心洞察：**
- **强情绪/高确定性** $\to$ $\kappa$ 大 $\to$ 分布尖锐 $\to$ 方差小
- **弱情绪/低确定性** $\to$ $\kappa$ 小 $\to$ 分布平坦 $\to$ 方差大

当 $\kappa \to \infty$，vMF 分布坍缩为狄拉克 delta 函数（点向量）。
当 $\kappa \to 0$，vMF 分布退化为均匀分布（完全模糊）。

---

## 2. Data Strategy: Soft Label Max-Norm (数据策略)

### 2.1 核心假设：强度即确定性

我们**不再使用 GPT-4 生成主观分数**（如 "Score 0.9"），而是让模型学习标签分布的**几何尖锐程度**。

**定义：** 情绪强度 $I_{raw}$ 为 Soft Label 分布的**无穷范数**（最大值），**但排除 Neutral 类别**：

$$I_{raw} = \max_{c \neq \text{neutral}} (\mathbf{y}_c)$$

其中 $\mathbf{y} \in \mathbb{R}^{28}$ 是归一化的情绪类别概率分布，$\sum_{i=1}^{28} y_i = 1$。

**关键修正（The Neutrality Paradox）：**
- 如果一句话的 Soft Label 是 `{"neutral": 0.9, "approval": 0.1}`
- 按原公式：$I_{raw} = 0.9 \to \kappa \approx 46$（高确定性，错误！）
- 修正后：$I_{raw} = 0.1 \to \kappa \approx 6$（低确定性，正确）

**原理：** 中性/平淡的句子应该在向量空间中占据"气态"位置（低κ），而不是"固态"统治地位。

### 2.2 几何解释

| $I_{raw}$ 值域 | 分布形态 | 认知解释 | 示例 |
|----------------|----------|----------|------|
| $\to 1.0$ | 极度尖锐（单峰突出） | 高确定性，情绪明确 | "I am furious!" |
| $\approx 0.5$ | 中等尖锐 | 中等确定性 | "I'm annoyed" |
| $\to 0.04$ | 接近均匀（$1/28$） | 低确定性，模棱两可 | "I don't know how I feel" |

### 2.3 数据生成 Prompt

使用以下 LLM Prompt 生成训练数据：

```
You are an emotion analysis expert. Analyze the emotional content of the following text.

Text: "{text}"

Task: Distribute exactly 1.0 probability mass across the 28 emotion categories below.

EMOTIONS = [
    # Positive
    "admiration", "amusement", "approval", "caring", "desire",
    "excitement", "gratitude", "joy", "love", "optimism",
    "pride", "relief",

    # Negative
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse",
    "sadness",

    # Ambiguous / Cognitive
    "confusion", "curiosity", "realization", "surprise",

    # Neutral
    "neutral"
]

Output Format (JSON only):
{
  "admiration": 0.05,
  "amusement": 0.0,
  "anger": 0.75,
  "annoyance": 0.10,
  ...
  "neutral": 0.05
}

Ensure all values are non-negative and sum to exactly 1.0.
```

### 2.4 数据格式 (V5: Multi-Target with Character Offsets)

**关键设计：解决 Token 对齐问题**

当句中存在重复词时，仅凭 `span_text` 无法确定标签对应哪个 Token。必须使用 **Character Offsets** 精确定位实体边界。

**V5 关键修正（LLM 字符计数幻觉）:**
- LLM 基于 Token 处理，无法精确计数字符
- **原则：** LLM 生成内容，Python 计算坐标
- LLM 只需输出 `span_text` 和 `soft_label`
- 脚本用 `text.find()` 或 `re.search()` 后处理计算 `char_start`, `char_end`

```json
{
  "text": "The cat ate the cat food.",
  "targets": [
    {
      "span_text": "cat",
      "soft_label": {"joy": 0.8, "neutral": 0.2}
    },
    {
      "span_text": "cat food",
      "soft_label": {"neutral": 0.9}
    }
  ]
}
```

**后处理 Pipeline (Python):**

```python
def postprocess_llm_output(text, llm_output):
    """
    LLM 输出后，使用 Python 计算字符坐标
    """
    for target in llm_output["targets"]:
        span_text = target["span_text"]
        # 使用 Python 精确定位
        idx = text.find(span_text)
        if idx == -1:
            # 尝试模糊匹配
            import re
            match = re.search(re.escape(span_text), text)
            if match:
                idx = match.start()
            else:
                raise ValueError(f"Span '{span_text}' not found in text")
        target["char_start"] = idx
        target["char_end"] = idx + len(span_text)
    return llm_output
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | string | 原始文本 |
| `targets` | list | 实体级别情感标注列表（一个句子可有多个目标） |
| `span_text` | string | 实体原文（由 LLM 生成） |
| `char_start` | int | **Python 后处理计算**：实体在原文中的字符起始位置（含） |
| `char_end` | int | **Python 后处理计算**：实体在原文中的字符结束位置（不含） |
| `soft_label` | dict | 28 类别的 Soft Label（稀疏表示，仅标注非零值） |

**Token 对齐逻辑：**

```python
# Tokenizer 返回 offsets_mapping
# "The cat ate" → [(0,3), (3,7), (7,11)]  # [The, cat, ate]
# Entity: "cat" with char_start=3, char_end=7

# 对齐公式：
entity_mask = (token_start < char_end) & (token_end > char_start) & attention_mask
```

---

## 3. Neural Architecture: Token-Level Bottlenecked Tri-Branch (V4 模型架构)

### 3.1 V4 核心变更：从 Pooling-First 到 Projection-First

**V3 (旧逻辑):**
```
BERT → [CLS] Token (768d) → Linear → μ (64d)
```
这是**句子级**的，只能获得整句的情感向量。

**V4 (新逻辑):**
```
BERT → Sequence Output (B, L, 768) → Linear → Sequence Vectors (B, L, 64) → Late Pooling (Entity Mask) → μ (64d)
```
这是**Token级**的，允许用户指定一个 `entity_mask` 来获取特定实体的情感向量。

### 3.2 架构概览 (V4: Project-then-Pool)

```
                    ┌─────────────────────────────────────┐
                    │        RoBERTa-base (768d)          │
                    └─────────────────────────────────────┘
                                      │ (Sequence Output: B, L, 768)
                    ┌─────────────────┴───────────────────┐
                    │                                     │
                    ▼                                     ▼
            ┌───────────────┐                     ┌───────────────┐
            │   Branch A    │                     │   Branch B    │
            │ (Dense Proj)  │                     │ (Dense Proj)  │
            └───────┬───────┘                     └───────┬───────┘
                    │ (B, L, 64)                          │ (B, L, 1)
                    ▼                                     ▼
            ┌───────────────┐                     ┌───────────────┐
            │ Mean Pooling  │                     │  Sum Pooling  │
            │ (Entity Mask) │                     │ (Entity Mask) │
            └───────┬───────┘                     └───────┬───────┘
                    │ μ (B, 64)                           │ Mass (B, 1)
                    ▼                                     ▼
            ┌───────────────┐                     ┌───────────────┐
            │   L2 Norm     │                     │    Scaling    │
            └───────┬───────┘                     │ κ = 1+α·Mass  │
                    │                             └───────────────┘
                    ▼
            ┌───────────────┐
            │   Branch C    │
            │  (Auxiliary)  │
            └───────┬───────┘
                    ▼
                 Logits
                (28d)
```

### 3.3 Branch A: Token-Level Semantic Core (语义核心)

**目标：** 为每个 Token 计算 vMF 分布的均值方向向量

**设计原理：** 基于 **Information Bottleneck (IB)** 理论 + **Late Pooling**，先对每个 Token 投影到 64d 空间，再根据需求聚合。

**网络结构：**

```
Last Hidden States (B, L, 768)
    │
    ▼
Linear(768 → 256)
    │
    ▼
GELU Activation
    │
    ▼
Linear(256 → 64)      ← Information Bottleneck per Token
    │
    ▼
┌─────────────────────────────────────┐
│     Token Vectors (B, L, 64)        │
│  (每个 Token 一个 64d 向量)          │
└─────────────────────────────────────┘
    │
    ├─────────────────────────────────┤
    │                                │
    ▼ (Entity Mask)                  ▼ (Sentence Mask)
Mean Pooling(entity_mask)      Mean Pooling(attention_mask)
    │                                │
    ▼                                ▼
μ_entity (64d)                   μ_sentence (64d)
    │                                │
    ▼                                ▼
L2 Normalization                 L2 Normalization
```

**Pooling 方式:**

| 推理模式 | Mask | Pooling | 输出 |
|----------|------|---------|------|
| **Sentence-Level** | `attention_mask` | Mean over valid tokens | 句子情感向量 |
| **Entity-Level** | `entity_mask` (用户指定 Span) | Mean over entity tokens | 实体情感向量 |

**关键点：**
- 每个 Token 都有一个 64d 向量
- Pooling **之后**才进行 L2 归一化：$\|\boldsymbol{\mu}\| = 1$
- 支持灵活的 Mask 传入，实现任意 Span 的情感分析

### 3.4 Branch B: Token-Level Physical Mass (物理质量)

**目标：** 为每个 Token 预测能量，聚合后得到专注度参数 $\kappa$

**设计原理：** 专注度（Concentration）是**强度量**，而非广度量

**关键修正（Length Bias 问题）：**
- ❌ **Sum Pooling：** 长句（30 tokens × 0.2 = 6.0）会大于短句（3 tokens × 0.9 = 2.7）
- ✅ **Max Pooling：** 衡量"最尖锐的情感爆发点"，与句子长度无关

**物理意义：** $\kappa$ 代表"最强烈的那个情感峰值"，而不是"情感的总量"

**网络结构：**

```
Last Hidden States (B, L, 768)
    │
    ▼
┌─────────┐
│ Energy  │
│ Proj    │
│(768→1)  │  ← 每个独立的能量预测
└─────────┘
    │
    ▼
Softplus(e_i)  ← 保证非负
    │
    ▼
┌─────────────────────────────────────┐
│     Token Energies (B, L, 1)        │
│  (每个 Token 一个能量值)            │
└─────────────────────────────────────┘
    │
    ├─────────────────────────────────┤
    │                                │
    ▼ (Entity Mask)                  ▼ (Sentence Mask)
Max Pooling(entity_mask)        Max Pooling(attention_mask)
    │                                │
    ▼                                ▼
E_entity (scalar)                 E_sentence (scalar)
    │                                │
    ▼                                ▼
κ = 1.0 + α × E                    κ = 1.0 + α × E
```

**物理意义：**
- **Max Pooling**：专注度是强度量（Intensive），与句子长度无关
- 实体的专注度 = 该实体所有 Token 中的**最大能量值**
- $\kappa = 1.0 + 50.0 \times \text{Max}(E_{entity})$

### 3.5 物理缩放参数与热力学解释

**物理缩放参数 $\alpha = 50.0$ 的理论依据：**

当 $I_{raw} \approx 1.0$（最高确定性）时：
- $\kappa_{target} = 1.0 + 50.0 \times 1.0 = 51$
- 对比学习温度 $\tau = 1/\kappa \approx 0.0196 \approx 0.02$

根据 InfoNCE 的理论分析，$\tau \approx 0.02$ 是**区分难负样本的最佳低温区间**，确保强情绪样本具有高特异性。

#### Thermodynamic Interpretation (热力学解释)

MATS 机制赋予了对比学习温度 $\tau$ 明确的物理意义，而非仅作为一个超参数：

| 热力学态 | $I_{raw}$ 范围 | $\kappa$ 值 | $\tau$ 值 | 系统行为 |
|----------|---------------|-------------|-----------|----------|
| **Solid State** (固态/结晶态) | $\to 1.0$ | 极大 ($\approx 50$) | $\to 0$ | 分布极度尖锐，类似于晶体结构。系统处于"低温低熵"状态，仅允许语义完全一致的样本匹配，实现高精度检索。 |
| **Gaseous State** (气态/高熵态) | $\to 0$ | 较小 ($\approx 1\text{--}10$) | 升高 | 分布平坦。系统处于"高温高熵"状态，容忍较大的语义距离，允许模糊匹配，实现高泛化性。 |

这种机制自适应地解决了**"Granularity-Specificity Trade-off"**（粒度-特异性权衡）难题：强情绪查询自动收缩匹配半径，弱情绪查询自动扩展匹配半径。

### 3.6 Branch C: Auxiliary Semantic Head (辅助语义头)

**目标：** 防止 64维瓶颈层在训练初期发生**语义坍塌** (Semantic Collapse)

**问题：** 仅使用对比学习损失时，瓶颈向量可能丢失细粒度的类别信息。

**解决方案：** 通过 KL 散度约束，强制 $\boldsymbol{\mu}$ 保留可恢复的情绪类别信息。

**网络结构：**

```
μ (64d)  ← 使用句子级 μ (Sentence-Level Pooling)
    │
    ▼
Linear(64 → 128)
    │
    ▼
GELU Activation
    │
    ▼
Linear(128 → 28)
    │
    ▼
aux_logits (28d)
```

**仅在训练时使用**，推理时可丢弃。

---

## 4. Training Strategy: Sample Flattening (训练策略)

### 4.1 从一对多到一对一：Sample Flattening

**问题：** 原始数据是一对多结构（1 个句子 → N 个实体目标），无法直接形成 GPU 并行 Batch。

**解决方案：** **Sample Flattening** — 在 Dataset 预处理阶段将 1 个句子拆解为 N 个训练样本。

```
原始数据：
"The cat played but the car broke." → 2 targets (cat=joy, car=anger)

展平后：
Sample 1: text="...", entity_mask=[cat], soft_label=[joy]
Sample 2: text="...", entity_mask=[car], soft_label=[anger]
```

**关键特性：**
- 同一个句子在 Dataset 中会出现多次（每个实体一次）
- 每个样本有自己独立的 `entity_mask`
- DataLoader 随机 Shuffle 后，Batch 内样本来自不同句子

### 4.2 训练流程

```python
for batch in dataloader:
    # batch['entity_mask']: (B, L) — 每个样本对应不同实体的 mask
    # batch['soft_label']: (B, 28) — 每个样本对应不同实体的 soft label

    # 单次前向传播（Supervised vMF-NCE）
    outputs = model(input_ids, attention_mask, entity_mask=batch['entity_mask'])

    # 计算损失（所有样本的 Loss 平均）
    loss = criterion(outputs, soft_labels)
    loss.backward()
```

**Loss 计算：**

$$L_{Total} = \frac{1}{M} \sum_{j=1}^{M} \left( L_{vMF}^{(j)} + \lambda_{Cal} \cdot L_{Cal}^{(j)} + \lambda_{Aux} \cdot L_{Aux}^{(j)} \right)$$

其中 $M$ 是 Batch 中展平后的实体样本总数。

### 4.3 Intensity 的实体级监督

**优势：** 现在的 $\kappa$ 直接由实体标签监督，而非从整句标签"猜测"。

| 示例 | Soft Label | Max-Norm (Intensity) | $\kappa$ (Target) |
|------|------------|---------------------|------------------|
| "The **cat** played" | joy=0.8, neutral=0.2 | 0.8 | $1 + 50 \times 0.8 = 41$ |
| "The **car** broke" | anger=0.9 | 0.9 | $1 + 50 \times 0.9 = 46$ |
| "I feel **meh**" | neutral=0.6 | 0.6 | $1 + 50 \times 0.6 = 31$ |

**对比 V3：**
- **V3:** 整句标签 → 模型需要"猜测"哪个 Token 承载情绪
- **V4:** 实体标签 → 直接监督，$\kappa$ 更准确

---

## 5. Training Objectives: Three-Part Loss (训练目标)

### 5.1 总损失函数

$$L_{Total} = L_{vMF} + \lambda_{Cal} \cdot L_{Cal} + \lambda_{Aux} \cdot L_{Aux}$$

推荐超参数：$\lambda_{Cal} = 0.1$, $\lambda_{Aux} = 0.05$

### 5.2 vMF-NCE Loss ($L_{vMF}$): 主损失

**原理：** 基于 vMF 分布假设的 Supervised Class-Prototype 对比学习

$$L_{vMF} = D_{KL}\left(\text{Softmax}\left(\kappa_i^{\text{detached}} \cdot \boldsymbol{\mu}_i^\top \boldsymbol{p}_c\right) \;\|\; \mathbf{y}_{soft}\right)$$

其中：
- $\boldsymbol{p}_c$ 是归一化的类别原型向量（$c \in \{1, \dots, 28\}$）
- $\kappa_i^{\text{detached}}$ 是**梯度截断**后的专注度参数，仅作为温度权重，不参与梯度更新
- 动态温度 $\tau_i = 1/\kappa_i^{\text{detached}}$

**关键设计：** 在 $L_{vMF}$ 中对 $\kappa$ 使用 `.detach()`，确保：
- $\boldsymbol{\mu}$ 和 $\boldsymbol{p}_c$ 由 $L_{vMF}$ 优化（语义方向）
- $\kappa$ 仅由 $L_{Cal}$ 优化（物理质量）

### 5.3 Calibration Loss ($L_{Cal}$): 校准损失

**目标：** 确保预测的 $\kappa_{pred}$ 与 Soft Label 的 Max-Norm 强度一致。

$$L_{Cal} = \text{MSE}\left(\kappa_{pred}, \kappa_{target}\right)$$

其中目标值为：

$$\kappa_{target} = 1.0 + 50.0 \times \max(\mathbf{y}_{soft})$$

### 5.4 Auxiliary Loss ($L_{Aux}$): 辅助损失

**目标：** 确保瓶颈向量 $\boldsymbol{\mu}$ 保留情绪类别信息。

$$L_{Aux} = D_{KL}\left(\text{Softmax}(\text{BranchC}(\boldsymbol{\mu})) \;\|\; \mathbf{y}_{soft}\right)$$

### 5.5 梯度流向设计 (Gradient Flow Considerations)

**问题：** 如果不正确处理梯度流向，训练可能出现以下问题：

1. **κ "作弊" 问题：**
   - 如果 $\kappa$ 的梯度可以流向 $L_{vMF}$，模型会发现：只要增大 $\kappa$ → 放大 logits → softmax 更尖锐 → KL Loss 更低
   - 这会导致 $\kappa$ 失去物理意义，不再是"强度"的度量，而变成一个用于降低损失的参数

2. **Prototype 模长膨胀：**
   - 如果不归一化 prototypes，模型可能通过增大原型向量的模长来增加 logits
   - 这违反了 vMF 分布的前提（所有向量应在超球面上）

**解决方案：**

| 设计 | 实现 | 目的 |
|------|------|------|
| **梯度截断** | `kappa.detach()` | 确保 $\kappa$ 仅由 $L_{Cal}$ 优化，$\boldsymbol{\mu}$ 和 $\boldsymbol{p}_c$ 由 $L_{vMF}$ 优化 |
| **原型归一化** | `F.normalize(self.prototypes)` | 确保原型向量在超球面上，防止模长膨胀 |

**梯度流向图：**

```
                    ┌─────────────────────────────────────┐
                    │           L_vMF (KL 散度)            │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    ▼                                     ▼
              ┌───────────┐                         ┌───────────┐
              │    μ      │                         │ Prototypes│
              │ (语义方向) │                         │ (类别中心) │
              └───────────┘                         └───────────┘
                    │                                     │
                    │     × (detach，无梯度)               │
                    ▼                                     ▼
              ┌───────────┐                         ┌───────────┐
              │    κ      │─────────────────────→   │   L_Cal   │
              │ (专注度)  │    仅被 L_Cal 优化        │  (MSE)    │
              └───────────┘                         └───────────┘
```

---

## 6. Complete PyTorch Implementation (V4)

### 6.1 Dataset Implementation

```python
import torch
from torch.utils.data import Dataset

class FineGrainedEmotionDataset(Dataset):
    """
    V4 Dataset with Character-Offset Alignment and Sample Flattening
    """
    def __init__(self, data, tokenizer, max_len=128):
        """
        Args:
            data: List[Dict] with keys: text, targets (list of entity annotations)
            tokenizer: HuggingFace tokenizer
            max_len: Max sequence length
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        # --- Sample Flattening: 1 sentence → N training samples ---
        for entry in data:
            text = entry['text']
            for target in entry['targets']:
                self.samples.append({
                    'text': text,
                    'char_start': target['char_start'],
                    'char_end': target['char_end'],
                    'soft_label': target['soft_label']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item['text']

        # 1. Tokenize with Offsets (Critical for alignment)
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True  # 返回字符偏移量
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        offsets = encoding['offset_mapping'].squeeze(0)  # (L, 2)

        # 2. Construct Entity Mask via Character Offsets
        c_start, c_end = item['char_start'], item['char_end']
        token_starts = offsets[:, 0]
        token_ends = offsets[:, 1]

        # Token 与 Entity 有交集 → True
        entity_mask = (token_starts < c_end) & (token_ends > c_start) & attention_mask.bool()
        entity_mask = entity_mask.float()

        # 3. Process Soft Label (Dict → 28d Vector)
        label_dict = item['soft_label']
        label_vector = torch.zeros(28)

        # Map emotion name to index (need EMOTION_INDEX mapping)
        for emotion, value in label_dict.items():
            if emotion in EMOTION_INDEX:
                label_vector[EMOTION_INDEX[emotion]] = value

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'entity_mask': entity_mask,  # (L,) — 当前实体的 mask
            'soft_label': label_vector   # (28,) — 当前实体的 soft label
        }


# Emotion name to index mapping
EMOTION_INDEX = {
    # Positive
    'admiration': 0, 'amusement': 1, 'approval': 2, 'caring': 3, 'desire': 4,
    'excitement': 5, 'gratitude': 6, 'joy': 7, 'love': 8, 'optimism': 9,
    'pride': 10, 'relief': 11,
    # Negative
    'anger': 12, 'annoyance': 13, 'disappointment': 14, 'disapproval': 15, 'disgust': 16,
    'embarrassment': 17, 'fear': 18, 'grief': 19, 'nervousness': 20, 'remorse': 21,
    'sadness': 22,
    # Ambiguous
    'confusion': 23, 'curiosity': 24, 'realization': 25, 'surprise': 26,
    # Neutral
    'neutral': 27
}
```

### 6.2 Model Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ProbabilisticGBERTV4(nn.Module):
    """
    Probabilistic G-BERT V4: Entity-Aware vMF Distribution for Text Embedding

    Architecture: Token-Level Bottlenecked Tri-Branch
    - Branch A: Token-Level Semantic Core → Late Pooling → μ (64d)
    - Branch B: Token-Level Energy → Sum Pooling → κ (1d)
    - Branch C: Auxiliary Semantic Head (28d logits)

    V4 Update: Projection-First architecture
    - Process each token independently to get (B, L, 64) vectors
    - Support both Sentence-Level and Entity-Level inference via flexible masking
    """

    def __init__(self, model_name='roberta-base', alpha_scale=50.0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = 768

        # --- Branch A: Token-Level Semantic Bottleneck (768 → 64 per token) ---
        self.semantic_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 64)
        )

        # --- Branch B: Token-Level Energy Projection (768 → 1 per token) ---
        self.energy_proj = nn.Linear(hidden_size, 1)
        self.alpha_scale = alpha_scale

        # --- Branch C: Auxiliary Semantic Head (64 → 28) ---
        self.aux_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 28)
        )

    def _pool_tokens(self, token_vectors: torch.Tensor, mask: torch.Tensor,
                     pooling_type: str = 'mean') -> torch.Tensor:
        """
        Pool token vectors using the specified mask.

        Args:
            token_vectors: (B, L, D) token-level vectors
            mask: (B, L) boolean mask (True = include in pooling)
            pooling_type: 'mean', 'sum', or 'max'

        Returns:
            (B, D) pooled vectors
        """
        # Expand mask for broadcasting: (B, L, 1)
        mask_expanded = mask.unsqueeze(-1).float()

        # 关键修正（Empty Mask 保护）：检查是否有全零的 mask
        valid_counts = mask.sum(dim=1)  # (B,)
        empty_mask = (valid_counts == 0)  # (B,) bool

        if pooling_type == 'mean':
            # Mean pooling: sum / count
            summed = torch.sum(token_vectors * mask_expanded, dim=1)
            count = torch.sum(mask_expanded, dim=1).clamp(min=1e-9)
            pooled = summed / count
            # 如果 mask 全为 0，使用 attention_mask 的结果作为 fallback
            # (这会在调用层处理，这里标记空 mask 的情况)
            return pooled
        elif pooling_type == 'sum':
            # Sum pooling
            return torch.sum(token_vectors * mask_expanded, dim=1)
        else:  # 'max'
            # Max pooling: 将 mask=0 的位置设为 -inf
            masked = torch.where(mask_expanded.bool(), token_vectors, torch.tensor(float('-inf'), device=token_vectors.device))
            pooled, _ = masked.max(dim=1)
            # 如果 mask 全为 0，max 会返回 -inf，需要处理
            # 使用 torch.where 将 -inf 替换为 0
            pooled = torch.where(empty_mask.unsqueeze(-1), torch.zeros_like(pooled), pooled)
            return pooled

    def forward(self, input_ids, attention_mask, entity_mask=None):
        """
        Forward pass with support for entity-level inference.

        Args:
            input_ids: (B, L) token ids
            attention_mask: (B, L) attention mask (1=valid, 0=pad)
            entity_mask: (B, L) optional entity mask (True=entity token)

        Returns:
            Dictionary containing:
            - mu: (B, 64) semantic direction (sentence-level by default)
            - mu_entity: (B, 64) entity semantic direction (if entity_mask provided)
            - kappa: (B, 1) concentration parameter
            - kappa_entity: (B, 1) entity concentration (if entity_mask provided)
            - aux_logits: (B, 28) auxiliary logits
        """
        # Backbone encoding
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, L, 768)

        # ===== Branch A: Token-Level Semantic Vectors =====
        # Project each token to 64d
        token_vectors = self.semantic_head(last_hidden)  # (B, L, 64)

        # Convert attention_mask to boolean (True = valid token)
        valid_mask = attention_mask.bool()  # (B, L)

        # Sentence-level pooling (default)
        mu = self._pool_tokens(token_vectors, valid_mask, pooling_type='mean')
        mu = F.normalize(mu, p=2, dim=1)  # (B, 64), ||μ|| = 1

        # Entity-level pooling (optional)
        mu_entity = None
        if entity_mask is not None:
            # 关键修正（Empty Mask 保护）：检查 entity_mask 是否全为 0
            empty_mask = (entity_mask.sum(dim=1) == 0)  # (B,) bool
            if empty_mask.any():
                import warnings
                warnings.warn(f"Empty entity_mask detected for {empty_mask.sum().item()} samples. "
                            f"Falling back to sentence-level pooling.")
                # 对空 mask 的样本，使用 valid_mask 替代
                entity_mask_safe = entity_mask.clone()
                entity_mask_safe[empty_mask] = valid_mask[empty_mask]
            else:
                entity_mask_safe = entity_mask

            mu_entity = self._pool_tokens(token_vectors, entity_mask_safe, pooling_type='mean')
            mu_entity = F.normalize(mu_entity, p=2, dim=1)

        # ===== Branch B: Token-Level Energy =====
        token_energies = F.softplus(self.energy_proj(last_hidden))  # (B, L, 1)

        # Sentence-level energy aggregation (max pooling - 专注度是强度量)
        energy_sentence = self._pool_tokens(token_energies, valid_mask, pooling_type='max')
        kappa = 1.0 + self.alpha_scale * energy_sentence  # (B, 1)

        # Entity-level energy aggregation (optional)
        kappa_entity = None
        if entity_mask is not None:
            # 使用相同的 entity_mask_safe（已在上面计算）
            if 'entity_mask_safe' in locals():
                energy_entity = self._pool_tokens(token_energies, entity_mask_safe, pooling_type='max')
                kappa_entity = 1.0 + self.alpha_scale * energy_entity
            else:
                # 如果没有进入 entity_mask 安全检查（上面的条件没触发），直接使用
                energy_entity = self._pool_tokens(token_energies, entity_mask, pooling_type='max')
                kappa_entity = 1.0 + self.alpha_scale * energy_entity

        # ===== Branch C: Auxiliary logits (uses sentence-level mu) =====
        aux_logits = self.aux_head(mu)  # (B, 28)

        result = {
            "mu": mu,
            "kappa": kappa,
            "aux_logits": aux_logits,
        }

        if entity_mask is not None:
            result.update({
                "mu_entity": mu_entity,
                "kappa_entity": kappa_entity,
            })

        return result


# ============================================================================
# Loss Functions
# ============================================================================

class SupervisedVMFNLoss(nn.Module):
    """
    Supervised vMF-NCE Loss with Class Prototypes.

    对比学习策略：拉近样本与其情感类别中心的距离

    关键设计：
    1. Prototypes 每次前向都归一化，确保在超球面上
    2. Kappa 使用 detach()，防止梯度泄露（L_vMF 只优化 mu 和 prototypes）
    """
    def __init__(self, num_emotions=28, embedding_dim=64):
        super().__init__()
        # 可学习的 Class Prototypes: 28 个情感类别的中心向量
        self.prototypes = nn.Parameter(torch.randn(num_emotions, embedding_dim))
        # L2 归一化初始化
        with torch.no_grad():
            self.prototypes.copy_(F.normalize(self.prototypes, p=2, dim=1))

    def forward(self, mu, kappa, soft_labels):
        """
        Args:
            mu: (B, 64) - 样本语义方向 (已归一化)
            kappa: (B, 1) - 样本专注度
            soft_labels: (B, 28) - Soft Label 分布

        Returns:
            L_vMF: vMF-NCE loss
        """
        # Step A: 归一化 Prototypes (防止模型通过增大模长"作弊")
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)

        # Step B: 计算 Cosine Similarity
        # mu: (B, 64), prototypes_norm: (28, 64) -> logits: (B, 28)
        logits = torch.matmul(mu, prototypes_norm.T)

        # Step C: 动态温度 (关键：Detach Kappa!)
        # L_vMF 只更新 mu 和 prototypes，kappa 由 L_Cal 单独优化
        kappa_fixed = kappa.detach()
        scaled_logits = logits * kappa_fixed  # 等价于 logits / (1/kappa)

        # Step D: Soft Label 作为目标分布 (多标签支持)
        log_probs = F.log_softmax(scaled_logits, dim=1)  # (B, 28)
        L_vMF = F.kl_div(log_probs, soft_labels, reduction='batchmean')

        return L_vMF


def calibration_loss(predicted_kappa, soft_labels, alpha_scale=50.0, neutral_idx=27):
    """
    Calibration Loss: align κ with Soft Label Max-Norm.

    关键修正：排除 Neutral 类别，防止中性句子获得高κ (The Neutrality Paradox)
    """
    # 排除 neutral 列 (假设 neutral 是最后一列，index=27)
    soft_labels_no_neutral = soft_labels[:, :neutral_idx]

    I_raw = torch.max(soft_labels_no_neutral, dim=1).values  # (B,)
    target_kappa = 1.0 + alpha_scale * I_raw                  # (B,)
    return F.mse_loss(predicted_kappa.squeeze(), target_kappa)


def auxiliary_loss(aux_logits, soft_labels):
    """Auxiliary Loss: KL divergence for semantic preservation."""
    log_pred = F.log_softmax(aux_logits, dim=1)
    return F.kl_div(log_pred, soft_labels, reduction='batchmean')


def total_loss(outputs, soft_labels, criterion, lambda_cal=0.1, lambda_aux=0.05):
    """Total Loss: L_vMF + λ_Cal * L_Cal + λ_Aux * L_Aux"""
    l_vmf = criterion(outputs['mu'], outputs['kappa'], soft_labels)
    l_cal = calibration_loss(outputs['kappa'], soft_labels)
    l_aux = auxiliary_loss(outputs['aux_logits'], soft_labels)

    return l_vmf + lambda_cal * l_cal + lambda_aux * l_aux
```

---

## 7. Training Pipeline (训练流程)

### 7.1 数据准备

```python
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # Load JSONL with fields: text, soft_label (28-dim list)
        self.data = [json.loads(line) for line in open(data_path)]

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = tokenizer(item['text'], max_length=128, padding='max_length')

        return {
            'input_ids': torch.tensor(encoded['input_ids']),
            'attention_mask': torch.tensor(encoded['attention_mask']),
            'soft_label': torch.tensor(item['soft_label'], dtype=torch.float32)
        }
```

### 7.2 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Backbone | `roberta-base` | 768d hidden size |
| Bottleneck Dim | 64 | Information Bottleneck |
| Effective Batch Size | 256 | 对比学习的目标 Batch 大小 |
| Physical Batch Size | 64 | 单次前向传播大小 (视显存调整) |
| Grad Accumulation | 4 | 梯度累积步数 (Physical × Accum = Effective) |
| Learning Rate (Backbone) | 2e-5 | 预训练模型小LR |
| Learning Rate (Heads) | 1e-4 | 新头可以用大LR |
| Weight Decay | 0.01 | |
| $\lambda_{Cal}$ | 0.1 | Calibration loss权重 |
| $\lambda_{Aux}$ | 0.05 | Auxiliary loss权重 |

### 7.3 监控指标

训练时需同时监控：

1. **Total Loss:** 整体收敛情况
2. **Average Kappa:** 预测的专注度分布
   - 预期：强样本 $\kappa \approx 40-50$，弱样本 $\kappa \approx 2-10$
3. **Auxiliary Accuracy:** 28分类准确率（仅用于监控，非最终目标）

---

## 8. Inference Strategy: One-Pass Mask-Based Inference (推理策略)

### 8.1 One-Pass, Mask-Based 推理

**核心原则：** 模型只运行一次 Forward Pass，通过改变 Mask 实现句子级或实体级分析。无需句子改写或分句。

#### 8.1.1 Sentence-Level Inference (句子级，默认)

**输入：** 仅文本
**Mask：** `attention_mask`（聚合所有有效 Token）

```python
def encode_sentence(text):
    encoded = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoded)  # entity_mask=None，默认使用 attention_mask
    return outputs['mu'], outputs['kappa']
```

**存储：** 向量数据库中存储 $\boldsymbol{\mu}$ (64d)

#### 8.1.2 Entity-Level Inference (实体级，V4 核心)

**输入：** Text + Entity Span (Character/Token Indices)
**Mask：** `entity_mask`（仅聚合指定 Span 的 Token）

```python
def encode_entity(text, entity_start, entity_end):
    """
    One-Pass Entity-Level Encoding

    Args:
        text: 输入文本
        entity_start, entity_end: 实体的 Token 位置索引
    """
    encoded = tokenizer(text, return_tensors='pt')

    # 创建 Entity Mask (B, L)
    entity_mask = torch.zeros_like(encoded['input_ids'], dtype=torch.bool)
    entity_mask[0, entity_start:entity_end] = True

    # 单次 Forward，使用 Entity Mask
    with torch.no_grad():
        outputs = model(**encoded, entity_mask=entity_mask)

    return outputs['mu'], outputs['kappa']  # 基于实体的 μ 和 κ
```

**示例：**
```python
text = "The movie was fantastic but acting terrible"

# 分析 "fantastic" 的情感
mu_fantastic, kappa_fantastic = encode_entity(text, entity_start=4, entity_end=5)
# κ ≈ 40-50，强烈积极情绪

# 分析 "terrible" 的情感
mu_terrible, kappa_terrible = encode_entity(text, entity_start=7, entity_end=8)
# κ ≈ 30-45，强烈消极情绪
```

**One-Pass 优势：**
- 无需重新 Tokenize 或改写句子
- 仅通过改变 Mask 即可提取同一句话中不同实体的情感
- 延迟低，适合实时应用

### 8.2 检索

```python
def search(query_text, query_entity_span=None, index=None, top_k=10):
    """
    One-Pass Entity-Aware Search

    Args:
        query_text: 查询文本
        query_entity_span: Optional (char_start, char_end) for entity-level search
        index: Vector database
        top_k: Return top-k results
    """
    encoded = tokenizer(query_text, return_tensors='pt')

    if query_entity_span is not None:
        # Entity-Level Search: Create entity mask from character offsets
        char_start, char_end = query_entity_span
        offsets = encoding['offset_mapping']  # Get offsets dynamically
        entity_mask = create_entity_mask_from_offsets(offsets, char_start, char_end)
        outputs = model(**encoded, entity_mask=entity_mask)
    else:
        # Sentence-Level Search: Use attention_mask
        outputs = model(**encoded)

    mu_q, kappa_q = outputs['mu'], outputs['kappa']

    # Vector similarity search
    candidates = index.search(mu_q, top_k=top_k)

    # Re-rank with mass-weighted score
    scores = kappa_q * torch.matmul(mu_q, candidates['mu'].T)

    return top_k_results


def create_entity_mask_from_offsets(offsets, char_start, char_end):
    """Helper: Convert character offsets to token mask"""
    token_starts = offsets[:, 0]
    token_ends = offsets[:, 1]
    return (token_starts < char_end) & (token_ends > char_start)
```

**核心公式：**

$$\text{Score}(q, d) = \kappa_q \cdot (\boldsymbol{\mu}_q^\top \boldsymbol{\mu}_d)$$

### 8.3 行为特性

| Query 类型 | $\kappa_q$ 值 | 检索行为 |
|------------|---------------|----------|
| 强情绪句子 (暴怒) | $\approx 50$ | 高敏感度，只返回语义最匹配的结果 |
| 弱情绪句子 (微烦) | $\approx 5$ | 低敏感度，返回多样化的结果 |
| 强情绪实体 (furious) | $\approx 40\text{--}50$ | 实体级高精度检索 |
| 弱情绪实体 (annoyed) | $\approx 5\text{--}15$ | 实体级多样化检索 |

### 8.4 Character Offset 对齐 (推理时)

推理时通常只有 Entity Text，需要先定位其字符位置：

```python
import re

def find_entity_offsets(text, entity_text):
    """Find character offsets of entity_text in text (first occurrence)"""
    match = re.search(re.escape(entity_text), text)
    if match:
        return match.start(), match.end()
    raise ValueError(f"Entity '{entity_text}' not found in text: {text}")


# 推理时使用
text = "The movie was fantastic but acting terrible"
entity_text = "fantastic"
char_start, char_end = find_entity_offsets(text, entity_text)
# 调用 search(query_text, query_entity_span=(char_start, char_end), ...)
```

---

## 9. Expected Contributions (学术贡献)

1. **理论贡献：** 将文本嵌入从欧氏空间的点假设扩展为超球面上的 vMF 分布假设，建立了"物理质量-统计专注度"的数学等价性。

2. **数据创新：** 提出 **Soft Label Max-Norm** 作为强度的几何定义，使模型学习可复现的分布几何量而非主观分数。

3. **架构创新 (V4)：** 提出 **Project-then-Pool** 架构，通过 Token-Level 瓶颈向量实现句子级和实体级的统一表示。

4. **训练策略创新 (V4)：** 提出 **Sample Flattening** 方法，将一对多的实体标注转换为可并行的一对一训练样本。

5. **工程创新 (V4)：** 使用 **Character Offsets** 解决 Token 对齐问题，实现精确的实体级情感标注。

6. **机制创新：** 提出了 **Mass-Adaptive Temperature Scaling (MATS)**，从 vMF 最大似然估计推导出自适应对比学习温度，并证明了 $\alpha = 50.0$ 的理论最优性。

7. **热力学解释：** MATS 机制将对比学习温度 $\tau$ 物理化。强情绪对应"固态/结晶态"（低温 $\tau \to 0$），仅允许精确匹配；弱情绪对应"气态"（高温），允许模糊匹配。

---

## Appendix A: GoEmotions 28 Categories

The emotion categories used for soft label generation (28 total):

```python
EMOTIONS = [
    # Positive (12)
    "admiration", "amusement", "approval", "caring", "desire",
    "excitement", "gratitude", "joy", "love", "optimism",
    "pride", "relief",

    # Negative (11)
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse",
    "sadness",

    # Ambiguous / Cognitive (4)
    "confusion", "curiosity", "realization", "surprise",

    # Neutral (1)
    "neutral"
]
```

**分布说明：**
- **Positive (12):** 积极情绪，通常伴随高唤醒度
- **Negative (11):** 消极情绪，包含愤怒、恐惧、悲伤等
- **Ambiguous/Cognitive (4):** 认知状态，可能为正可能为负
- **Neutral (1):** 中性状态，作为基准类别

---

## Appendix B: V3 vs V4 Architecture Comparison

| 特性 | V3 (Sentence-Level) | V4 (Entity-Aware) |
|------|---------------------|-------------------|
| **Branch A 输入** | [CLS] Token (768d) | Full Sequence (B, L, 768) |
| **Branch A 输出** | μ (64d) | Token Vectors (B, L, 64) → Pooled μ |
| **Pooling 时机** | Early (先取 CLS) | Late (先投影后聚合) |
| **Branch B 聚合** | Attention Weighted Sum | Direct Sum (基于 Mask) |
| **推理模式** | 仅 Sentence-Level | Sentence + Entity 双模式 |
| **实体级分析** | 不支持 | 原生支持 |
| **训练需求** | 句子级 Soft Label | 句子级 Soft Label (不变) |

---

*Document Version: V4 (Entity-Aware & One-Pass Edition)*
*Last Updated: 2025*
