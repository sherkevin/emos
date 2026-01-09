# Probabilistic G-BERT: System Architecture V3 (NIPS Edition)

## Abstract

现有的文本嵌入模型（如 BERT, E5）通常将语义映射为欧氏空间中的**确定性点向量**。这种**各向同性（Isotropic）**的假设忽略了人类记忆与情感的一个核心特征：**"强度即确定性" (Intensity implies Certainty)**。

为此，我们提出 **Probabilistic G-BERT**。不同于学习静态向量，我们将每个文本建模为超球面上的 **Von Mises-Fisher (vMF)** 概率分布。

1. **理论基础：** 从欧氏空间的点假设转向超球面上的**概率分布**假设。
2. **数据策略：** 引入 **Soft Label Max-Norm**，将"强度"定义为标签分布的几何尖锐程度。
3. **架构创新：** 设计了 **Bottlenecked Tri-Branch** 网络，同时学习语义方向、物理质量和辅助语义约束。
4. **训练机制：** 提出了 **Mass-Adaptive Temperature Scaling (MATS)**，从 vMF 最大似然估计推导动态对比学习温度。

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

**定义：** 情绪强度 $I_{raw}$ 为 Soft Label 分布的**无穷范数**（最大值）：

$$I_{raw} = \|\mathbf{y}\|_\infty = \max(\text{Soft\_Labels})$$

其中 $\mathbf{y} \in \mathbb{R}^{28}$ 是归一化的情绪类别概率分布，$\sum_{i=1}^{28} y_i = 1$。

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

### 2.4 数据格式

```json
{
  "text": "I am absolutely furious right now!",
  "soft_label": [0.01, 0.02, 0.85, 0.05, ...],  // 28-dim probability vector
  "intensity": 0.85  // Max-Norm, computed as max(soft_label)
}
```

---

## 3. Neural Architecture: Bottlenecked Tri-Branch (模型架构)

### 3.1 架构概览

```
                    ┌─────────────────────────────────────┐
                    │        RoBERTa-base (768d)          │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │                                     │
                    ▼                                     ▼
            ┌───────────────┐                     ┌───────────────┐
            │   Branch A    │                     │   Branch B    │
            │  (Semantic)   │                     │    (Mass)     │
            └───────┬───────┘                     └───────┬───────┘
                    │                                     │
                    ▼                                     ▼
            ┌───────────────┐                     ┌───────────────┐
            │       μ       │                     │       κ       │
            │     (64d)     │                     │     (1d)      │
            └───────┬───────┘                     └───────────────┘
                    │
                    ▼
            ┌───────────────┐
            │   Branch C    │
            │  (Auxiliary)  │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │    logits     │
            │     (28d)     │
            └───────────────┘
```

### 3.2 Branch A: Semantic Core (语义核心)

**目标：** 提取 vMF 分布的均值方向 $\boldsymbol{\mu}$

**设计原理：** 基于 **Information Bottleneck (IB)** 理论，通过强制降维迫使模型丢弃句法噪声，只保留核心语义。

**网络结构：**

```
[CLS] Token (768d)
    │
    ▼
Linear(768 → 256)
    │
    ▼
GELU Activation
    │
    ▼
Linear(256 → 64)      ← Information Bottleneck
    │
    ▼
L2 Normalization
    │
    ▼
μ ∈ 𝑆⁶³ (64d unit vector)
```

**关键点：**
- 输出维度严格限制为 **64维**
- 输出必须 L2 归一化：$\|\boldsymbol{\mu}\| = 1$

### 3.3 Branch B: Physical Mass (物理质量)

**目标：** 预测 vMF 分布的专注度参数 $\kappa$

**设计原理：** 通过 **Gravitational Attention** 机制，聚合 Token 级别的能量，模拟物理质量的形成过程。

**网络结构：**

```
Last Hidden States (B, L, 768)
    │
    ├─────────────────┬─────────────────┐
    ▼                 ▼
┌─────────┐     ┌─────────┐
│ Energy  │     │ Attn    │
│ Proj    │     │ Proj    │
│(768→1)  │     │(768→1)  │
└─────────┘     └─────────┘
    │                 │
    ▼                 ▼
Softplus(e_i)    Softmax(α_i)    + Mask(padding)
    │                 │
    └─────────────────┴─────────────────┘
                      │
                      ▼
              Weighted Sum: Σ α_i · e_i
                      │
                      ▼
                  mass_raw
                      │
                      ▼
         κ = 1.0 + α × mass_raw
         (α = 50.0, see scaling theory)
```

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

### 3.4 Branch C: Auxiliary Semantic Head (辅助语义头)

**目标：** 防止 64维瓶颈层在训练初期发生**语义坍塌** (Semantic Collapse)

**问题：** 仅使用对比学习损失时，瓶颈向量可能丢失细粒度的类别信息。

**解决方案：** 通过 KL 散度约束，强制 $\boldsymbol{\mu}$ 保留可恢复的情绪类别信息。

**网络结构：**

```
μ (64d)
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

## 4. Training Objectives: Three-Part Loss (训练目标)

### 4.1 总损失函数

$$L_{Total} = L_{vMF} + \lambda_{Cal} \cdot L_{Cal} + \lambda_{Aux} \cdot L_{Aux}$$

推荐超参数：$\lambda_{Cal} = 0.1$, $\lambda_{Aux} = 0.05$

### 4.2 vMF-NCE Loss ($L_{vMF}$): 主损失

**原理：** 基于 vMF 分布假设的 InfoNCE 对比学习

$$L_{vMF} = -\log \frac{\exp(\boldsymbol{\mu}_i^\top \boldsymbol{\mu}_{+} / \tau_i)}{\sum_{k=1}^{N} \exp(\boldsymbol{\mu}_i^\top \boldsymbol{\mu}_k / \tau_i)}$$

其中**动态温度**定义为：

$$\tau_i = \frac{1}{\kappa_i} = \frac{1}{1.0 + 50.0 \times I_{raw}^{(i)}}$$

**PyTorch 实现：**

```python
def vmf_nce_loss(mu, kappa, labels):
    """
    Args:
        mu: (B, 64) L2-normalized semantic directions
        kappa: (B, 1) concentration parameters
        labels: (B,) positive sample indices
    """
    # Compute cosine similarity matrix
    logits = torch.matmul(mu, mu.T)  # (B, B)

    # Dynamic temperature: tau = 1 / kappa
    tau = 1.0 / (kappa + 1e-6)  # (B, 1)

    # Apply MATS: scale logits by concentration
    scaled_logits = logits / tau  # (B, B) / (B, 1) → (B, B)

    # Standard cross-entropy
    loss = F.cross_entropy(scaled_logits, labels)
    return loss
```

### 4.3 Calibration Loss ($L_{Cal}$): 校准损失

**目标：** 确保预测的 $\kappa_{pred}$ 与 Soft Label 的 Max-Norm 强度一致。

$$L_{Cal} = \text{MSE}\left(\kappa_{pred}, \kappa_{target}\right)$$

其中目标值为：

$$\kappa_{target} = 1.0 + 50.0 \times \max(\mathbf{y}_{soft})$$

**PyTorch 实现：**

```python
def calibration_loss(predicted_kappa, soft_labels):
    """
    Args:
        predicted_kappa: (B, 1) model output
        soft_labels: (B, 28) ground-truth probability distributions
    """
    # Intensity as Max-Norm of Soft Label
    I_raw = torch.max(soft_labels, dim=1).values  # (B,)

    # Target: κ = 1.0 + 50.0 × I_raw
    target_kappa = 1.0 + 50.0 * I_raw  # (B,)

    # MSE loss
    loss = F.mse_loss(predicted_kappa.squeeze(), target_kappa)
    return loss
```

### 4.4 Auxiliary Loss ($L_{Aux}$): 辅助损失

**目标：** 确保瓶颈向量 $\boldsymbol{\mu}$ 保留情绪类别信息。

$$L_{Aux} = D_{KL}\left(\text{Softmax}(\text{BranchC}(\boldsymbol{\mu})) \;\|\; \mathbf{y}_{soft}\right)$$

**PyTorch 实现：**

```python
def auxiliary_loss(aux_logits, soft_labels):
    """
    Args:
        aux_logits: (B, 28) raw output from Branch C
        soft_labels: (B, 28) ground-truth probability distributions
    """
    log_pred = F.log_softmax(aux_logits, dim=1)
    loss = F.kl_div(log_pred, soft_labels, reduction='batchmean')
    return loss
```

---

## 5. Complete PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ProbabilisticGBERT(nn.Module):
    """
    Probabilistic G-BERT: vMF Distribution for Text Embedding

    Architecture: Bottlenecked Tri-Branch
    - Branch A: Semantic Core (64d unit vector)
    - Branch B: Physical Mass (concentration κ)
    - Branch C: Auxiliary Semantic Head (28d logits)
    """

    def __init__(self, model_name='roberta-base', alpha_scale=50.0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = 768

        # --- Branch A: Semantic Bottleneck (768 → 64) ---
        self.semantic_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 64)
        )

        # --- Branch B: Gravitational Attention ---
        self.energy_proj = nn.Linear(hidden_size, 1)      # Token energy
        self.attn_proj = nn.Linear(hidden_size, 1)        # Attention weights
        self.alpha_scale = alpha_scale                    # Scaling factor

        # --- Branch C: Auxiliary Semantic Head (64 → 28) ---
        self.aux_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 28)
        )

    def forward(self, input_ids, attention_mask):
        # Backbone encoding
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, L, 768)
        cls_emb = last_hidden[:, 0, :]           # (B, 768)

        # Branch A: Semantic Direction μ
        raw_vec = self.semantic_head(cls_emb)
        mu = F.normalize(raw_vec, p=2, dim=1)    # (B, 64), ||μ|| = 1

        # Branch B: Concentration κ via Gravitational Attention
        token_energies = F.softplus(self.energy_proj(last_hidden))  # (B, L, 1)
        attn_scores = self.attn_proj(last_hidden)                   # (B, L, 1)

        # Mask padding tokens
        mask = attention_mask.unsqueeze(-1)                          # (B, L, 1)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)                # (B, L, 1)

        # Aggregated mass
        mass = torch.sum(attn_weights * token_energies, dim=1)      # (B, 1)

        # Physical scaling: κ = 1.0 + α × mass
        kappa = 1.0 + self.alpha_scale * mass                       # (B, 1)

        # Branch C: Auxiliary logits
        aux_logits = self.aux_head(mu)                              # (B, 28)

        return {
            "mu": mu,              # (B, 64) - semantic direction
            "kappa": kappa,        # (B, 1)  - concentration
            "mass": mass,          # (B, 1)  - for visualization
            "aux_logits": aux_logits  # (B, 28) - for L_Aux
        }


def vmf_nce_loss(mu, kappa, labels):
    """vMF-NCE Loss with adaptive temperature."""
    logits = torch.matmul(mu, mu.T)               # (B, B)
    tau = 1.0 / (kappa + 1e-6)                    # (B, 1)
    scaled_logits = logits / tau                  # (B, B)
    return F.cross_entropy(scaled_logits, labels)


def calibration_loss(predicted_kappa, soft_labels):
    """Calibration Loss: align κ with Soft Label Max-Norm."""
    I_raw = torch.max(soft_labels, dim=1).values           # (B,)
    target_kappa = 1.0 + 50.0 * I_raw                      # (B,)
    return F.mse_loss(predicted_kappa.squeeze(), target_kappa)


def auxiliary_loss(aux_logits, soft_labels):
    """Auxiliary Loss: KL divergence for semantic preservation."""
    log_pred = F.log_softmax(aux_logits, dim=1)
    return F.kl_div(log_pred, soft_labels, reduction='batchmean')


def total_loss(outputs, soft_labels, labels, lambda_cal=0.1, lambda_aux=0.05):
    """Total Loss: L_vMF + λ_Cal * L_Cal + λ_Aux * L_Aux"""
    l_vmf = vmf_nce_loss(outputs['mu'], outputs['kappa'], labels)
    l_cal = calibration_loss(outputs['kappa'], soft_labels)
    l_aux = auxiliary_loss(outputs['aux_logits'], soft_labels)

    return l_vmf + lambda_cal * l_cal + lambda_aux * l_aux
```

---

## 6. Training Pipeline (训练流程)

### 6.1 数据准备

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

### 6.2 训练配置

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

### 6.3 监控指标

训练时需同时监控：

1. **Total Loss:** 整体收敛情况
2. **Average Kappa:** 预测的专注度分布
   - 预期：强样本 $\kappa \approx 40-50$，弱样本 $\kappa \approx 2-10$
3. **Auxiliary Accuracy:** 28分类准确率（仅用于监控，非最终目标）

---

## 7. Inference Strategy (推理策略)

### 7.1 存储

向量数据库中**仅存储 $\boldsymbol{\mu}$ (64d)**。

### 7.2 检索

```python
def search(query_text, index, top_k=10):
    # 1. Encode query
    outputs = model(query_text)
    mu_q = outputs['mu']        # (1, 64)
    kappa_q = outputs['kappa']  # (1, 1)

    # 2. Retrieve candidates (vector similarity)
    candidates = index.search(mu_q, top_k=100)

    # 3. Re-rank with mass-weighted score
    scores = kappa_q * torch.matmul(mu_q, candidates['mu'].T)

    return top_k_results
```

**核心公式：**

$$\text{Score}(q, d) = \kappa_q \cdot (\boldsymbol{\mu}_q^\top \boldsymbol{\mu}_d)$$

### 7.3 行为特性

| Query 类型 | $\kappa_q$ 值 | 检索行为 |
|------------|---------------|----------|
| 强情绪 (暴怒) | $\approx 50$ | 高敏感度，只返回语义最匹配的结果 |
| 弱情绪 (微烦) | $\approx 5$ | 低敏感度，返回多样化的结果 |

---

## 8. Expected Contributions (学术贡献)

1. **理论贡献：** 将文本嵌入从欧氏空间的点假设扩展为超球面上的 vMF 分布假设，建立了"物理质量-统计专注度"的数学等价性。

2. **数据创新：** 提出 **Soft Label Max-Norm** 作为强度的几何定义，使模型学习可复现的分布几何量而非主观分数。

3. **架构创新：** 设计了 **Bottlenecked Tri-Branch** 结构，通过信息瓶颈提取核心语义，同时通过辅助损失防止语义坍塌。

4. **机制创新：** 提出了 **Mass-Adaptive Temperature Scaling (MATS)**，从 vMF 最大似然估计推导出自适应对比学习温度，并证明了 $\alpha = 50.0$ 的理论最优性。

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

*Document Version: V3 (NIPS Edition)*
*Last Updated: 2025*
