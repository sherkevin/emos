# DESGIN：Probabilistic G-BERT: 工程实现设计文档 (Implementation Design Doc)

## 1. 项目概览 (Project Overview)

* **项目名称:** `probabilistic_gbert`
* **核心架构:** Bottlenecked Tri-Branch (RoBERTa → Token-Level Projection → Masked Pooling → 64d μ / 1d κ)
* **架构流向:** **串行结构** - Branch C 接在 Branch A 的 Pooled 输出端，形成信息瓶颈约束
* **训练目标:** 3-Part Loss (vMF-NCE + Calibration + Auxiliary)
* **对比学习策略:** **Supervised vMF-NCE** - Class-Prototype 监督信号，拉近样本与情感类别中心的距离
* **实体支持 (V4 新增):** Token 粒度 Entity Masking + Sample Flattening + Character Offsets，支持实体级语义提取
* **硬件策略:** 优先使用 CUDA (GPU)，若不可用自动回退至 CPU。
* **开发框架:** Python 3.9+, PyTorch 2.0+, Transformers

---

## 2. 关键决策矩阵 (Decision Matrix)

| # | 问题 | 最终决策 (Final Decision) |
|---|------|---------------------------|
| 1 | 对比学习策略 | **Supervised vMF-NCE** - Class-Prototype 监督信号，拉近样本与情感类别中心的距离 |
| 2 | 数据生成 | **优先复用 GoEmotions Raw Votes**；LLM 脚本用 gpt-4o-mini 备用 |
| 3 | LR Scheduler | **Linear Warmup (10%) + Decay** |
| 4 | Checkpoint | 保存 **best_model.pt** (权重) 和 **last.pt** (全状态) |
| 5 | Logging | **Console (Standard) + WandB (If available)** |
| 6 | Masking (V4) | **Token-Level Projection** + **Masked Pooling**；支持 `entity_mask` 实体提取 |
| 7 | 推理功能 | **单句分析模式** (entity_mask=None) + **实体提取模式** (entity_mask 指定) |
| 8 | Batch Size | 显存允许下的最大值 (如 64) + 梯度累积 (至 256) |
| 9 | Pooling 策略 (V4) | Branch A: **Mean Pool**；Branch B: **Max Pool** (专注度是强度量) |
| 10 | Token 对齐 (V4) | **Character Offsets** - 使用 `char_start`, `char_end` 精确定位，避免重复词歧义 |
| 11 | 数据采样 (V4) | **Sample Flattening** - `__init__` 时展平所有 targets，1 句 N 实体 → N 个训练样本 |
| 12 | 训练监督 (V4) | **Supervised Multi-Granularity** - 直接在实体级别计算 Loss，非 Zero-Shot |

---

## 3. 项目目录结构 (Directory Structure)

```text
probabilistic_gbert/
├── data/                        # 数据存储
│   ├── raw/                     # 原始数据 (GoEmotions)
│   ├── processed/               # 处理后的 JSONL (带 Soft Labels)
│   └── generate_data.py         # LLM 数据生成脚本 (gpt-4o-mini, 备用)
├── src/                         # 源代码
│   ├── __init__.py
│   ├── config.py                # 全局配置参数 (Hyperparameters)
│   ├── dataset.py               # 数据加载与处理 (GoEmotions raw votes)
│   ├── model.py                 # PyTorch 模型定义 (Tri-Branch 串行结构)
│   ├── loss.py                  # 损失函数定义 (Supervised vMF-NCE + Calibration + Auxiliary)
│   └── utils.py                 # 工具函数 (Logger, Metrics, Device)
├── checkpoints/                 # 模型保存路径
│   ├── best_model.pt            # 最优模型权重 (用于推理)
│   └── last_checkpoint.pt       # 最新完整检查点 (用于断点续训)
├── train.py                     # 训练主入口
├── inference.py                 # 交互式推理 Demo
├── requirements.txt             # 依赖包
└── README.md                    # 项目说明
```

---

## 4. 模块功能详细描述 (Module Specifications)

### 4.1 `src/config.py` (配置中心)

管理所有超参数，确保实验可复现。

* **类:** `Config` (使用 `dataclass`)

* **关键参数:**

| 参数 | 值 | 说明 |
|------|-----|------|
| `model_name` | "roberta-base" | 预训练模型 |
| `embedding_dim` | 64 | Bottleneck 维度 |
| `num_emotions` | 28 | GoEmotions 类别数 |
| `alpha_scale` | 50.0 | 物理缩放系数 (κ = 1.0 + α × mass) |
| `max_length` | 128 | 文本最大长度 |
| `physical_batch_size` | 64 | 单次前向传播大小 (视显存调整) |
| `effective_batch_size` | 256 | 对比学习目标 Batch 大小 (Physical × Accum) |
| `grad_accum_steps` | 4 | 梯度累积步数 (64 × 4 = 256) |
| `lr_backbone` | 2e-5 | Backbone 学习率 |
| `lr_heads` | 1e-4 | 分支学习率 |
| `epochs` | 5 | 训练轮数 |
| `warmup_ratio` | 0.1 | Warmup 占总步数的比例 |
| `lambda_cal` | 0.1 | 校准损失权重 |
| `lambda_aux` | 0.05 | 辅助损失权重 |
| `patience` | 3 | Early Stopping 容忍轮数 |
| `device` | 动态获取 | "cuda" or "cpu" |

### 4.2 `src/dataset.py` (数据管道)

**数据来源优先级:** GoEmotions Raw Annotations → LLM 生成

#### V4 数据格式与字符对齐 (Critical)

**关键变更：解决 Token 对齐问题**

当句中存在重复词时，仅凭 `span_text` 无法确定标签对应哪个 Token。必须使用 **Character Offsets** 精确定位实体边界。

**JSONL 格式 (多目标 Span with Character Offsets):**
```json
{
  "text": "The cat ate the cat food.",
  "targets": [
    {
      "span_text": "cat",
      "char_start": 4,
      "char_end": 7,
      "soft_label": {"joy": 0.8, "neutral": 0.2}
    },
    {
      "span_text": "cat food",
      "char_start": 16,
      "char_end": 24,
      "soft_label": {"neutral": 0.9}
    }
  ]
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | string | 原始文本 |
| `targets` | list | 实体级别情感标注列表（一个句子可有多个目标） |
| `span_text` | string | 实体原文（仅供参考，不用于对齐） |
| `char_start` | int | 实体在原文中的**字符起始位置**（含） |
| `char_end` | int | 实体在原文中的**字符结束位置**（不含） |
| `soft_label` | dict | 28 类别的 Soft Label（稀疏表示，仅标注非零值） |

#### Sample Flattening 策略

**问题：** 原始数据是一对多结构（1 个句子 → N 个实体目标），无法直接形成 GPU 并行 Batch。

**解决方案：** **Sample Flattening** — 在 Dataset 预处理阶段将 1 个句子拆解为 N 个训练样本。

```
原始数据：
"The cat played but the car broke." → 2 targets (cat=joy, car=anger)

展平后：
Sample 1: text="...", entity_mask=[cat], soft_label=[joy]
Sample 2: text="...", entity_mask=[car], soft_label=[anger]
```

#### GoEmotions 处理逻辑
GoEmotions 原始数据包含多名标注者的投票（如 3/10 人标记为 Anger），这本身就是 Soft Label：
```python
# 假设原始格式: {text: "...", labels: [0, 0, 3, 1, ...]} (每个类别的标注人数)
# 归一化为概率分布
soft_label = labels / labels.sum()
```

* **类:** `FineGrainedEmotionDataset(Dataset)`

* **`__init__(data_path, tokenizer, max_len)`:`
  * 保存 tokenizer 和 max_len
  * **Sample Flattening:** 遍历所有 entry 的 targets，将 1 句 N 实体 展平为 N 个独立样本

* **`_create_entity_mask(text, span_text, encoding)`:` - **字符对齐核心函数**
  ```python
  def _create_entity_mask(self, text, span_text, encoding):
      """
      通过字符级对齐生成 entity_mask

      Args:
          text: 原始文本
          span_text: 实体文本 (如 "cat")
          encoding: tokenizer 的输出，包含 offset_mapping

      Returns:
          entity_mask: (L,) Tensor, 1 for entity tokens, 0 otherwise
      """
      # 使用显式的 character offsets（不再需要 text.find()）
      c_start, c_end = item['char_start'], item['char_end']
      token_starts = offsets[:, 0]
      token_ends = offsets[:, 1]

      # Token 与 Entity 有交集 → True
      entity_mask = (token_starts < c_end) & (token_ends > c_start) & attention_mask.bool()
      return entity_mask.float()
  ```

* **`__getitem__(idx)`:**
  1. 加载数据项 `item = self.data[idx]`
  2. **Tokenize:** 必须开启 `return_offsets_mapping=True`
     ```python
     encoding = self.tokenizer(
         item['text'],
         max_length=self.max_len,
         padding='max_length',
         truncation=True,
         return_offsets_mapping=True  # Critical for char-to-token alignment
     )
     ```
  3. **Span 采样:**
     ```python
     # Sample 已在 __init__ 中展平，直接使用 item['char_start'], item['char_end']
     c_start, c_end = item['char_start'], item['char_end']
     ```
  4. **生成 entity_mask:**
     ```python
     # Construct Entity Mask via Character Offsets
     offsets = encoding['offset_mapping'].squeeze(0)  # (L, 2)
     token_starts = offsets[:, 0]
     token_ends = offsets[:, 1]

     # Token 与 Entity 有交集 → True
     entity_mask = (token_starts < c_end) & (token_ends > c_start) & attention_mask.bool()
     entity_mask = entity_mask.float()
     ```
  5. **处理 Soft Label (Dict → 28d Vector):**
     ```python
     label_dict = item['soft_label']
     label_vector = torch.zeros(28)

     # Map emotion name to index (need EMOTION_INDEX mapping)
     for emotion, value in label_dict.items():
         if emotion in EMOTION_INDEX:
             label_vector[EMOTION_INDEX[emotion]] = value
     ```
  6. 返回字典:
     ```python
     {
         'input_ids': torch.tensor(encoding['input_ids']),
         'attention_mask': torch.tensor(encoding['attention_mask']),
         'entity_mask': entity_mask,  # (L,) — 当前实体的 mask
         'soft_label': label_vector   # (28,) — 当前实体的 soft label
     }
     ```

* **Token 对齐公式:**
  ```python
  entity_mask = (token_start < char_end) & (token_end > char_start) & attention_mask
  ```

* **函数:** `create_dataloaders(config)`
  * 返回 Train/Val DataLoader。
  * **注意:** 使用 Random Shuffle 即可，Supervised vMF-NCE 不需要特殊分组。

#### LLM 生成脚本 (`data/generate_data.py`)

* **用途:** 为非 GoEmotions 数据源生成带 Span 的 Soft Labels
* **模型:** gpt-4o-mini (成本低，质量足够)
* **API Key:** `os.getenv("OPENAI_API_KEY")`

**关键修正（LLM 字符计数幻觉）:**
- LLM 基于 Token 处理，无法精确计数字符
- **原则：** LLM 生成内容，Python 计算坐标
- LLM 只需输出 `span_text` 和 `soft_label`
- 脚本用 `text.find()` 或 `re.search()` 后处理计算 `char_start`, `char_end`

* **Prompt (V5 修正版):** 要求 LLM 输出 `span_text` 和 `soft_label`（不输出字符坐标）

```python
PROMPT_V5 = """
You are an emotion analysis expert. Analyze the emotional content of entities in the following text.

Text: "{text}"

Task: Identify up to 3 key entities/phrases and distribute 1.0 probability mass across the 28 emotion categories for each.

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
{{
  "targets": [
    {{
      "span_text": "exact phrase from text",
      "soft_label": {{"admiration": 0.05, ...}}
    }},
    ...
  ]
}}

IMPORTANT:
- span_text must exactly match the text (use word boundaries)
- Do NOT output character indices - they will be computed automatically
"""

# 后处理 Pipeline (Python)
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

### 4.3 `src/model.py` (模型架构) - **V4: Entity-Aware**

实现 System V4 的三分支**串行**结构，支持 Token 粒度的实体感知操作。

**架构流向:**
```
RoBERTa (768d) → last_hidden_state (B, L, 768)
    ├─→ Branch A (Semantic)   → semantic_seq (B, L, 64) → Masked Mean Pool → μ (64d)
    ├─→ Branch B (Mass)       → energy_seq (B, L, 1)   → Masked Sum Pool  → κ (1d)
    └─→ Branch C (Auxiliary)  → μ (64d) → logits (28d)
```

* **类:** `ProbabilisticGBERT(nn.Module)`

* **`__init__(config)`:**
  * 初始化 Backbone (RoBERTa-base)。
  * **Branch A:** `Sequential(Linear(768→256), GELU, Linear(256→64))`。
  * **Branch B:** `Sequential(Linear(768→128), GELU, Linear(128→1))` + `Softplus`。
  * **Branch C:** `Sequential(Linear(64→128), GELU, Linear(128→28))`。
  * `alpha_scale`: 物理缩放系数（默认 50.0）。

* **`forward(input_ids, attention_mask, entity_mask=None)`:**
  ```python
  """
  Args:
      input_ids: (B, L) Token IDs
      attention_mask: (B, L) 1=Valid, 0=Padding
      entity_mask: (B, L) - 1 for tokens in the entity, 0 otherwise.
                   If None, defaults to attention_mask (whole sentence).

  Returns:
      dict with mu (B, 64), kappa (B, 1), aux_logits (B, 28), mass (B, 1)
  """
  ```

  **内部处理流程:**

  1. **Backbone Encoding:**
     ```python
     outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
     last_hidden = outputs.last_hidden_state  # (B, L, 768)
     ```

  2. **Dense Projection (应用于整个序列):**
     ```python
     # Branch A: Semantic projection to token-level
     semantic_seq = self.semantic_head(last_hidden)  # (B, L, 64)

     # Branch B: Energy projection to token-level
     energy_seq = F.softplus(self.energy_head(last_hidden))  # (B, L, 1)
     ```

  3. **Masking Strategy:**
     ```python
     # Determine pooling mask
     if entity_mask is None:
         pool_mask = attention_mask  # (B, L) - whole sentence
     else:
         # Entity mode: must still respect padding
         pool_mask = entity_mask * attention_mask  # (B, L)

     pool_mask = pool_mask.unsqueeze(-1)  # (B, L, 1) for broadcasting
     ```

  4. **Manual Pooling (关键步骤):**

     **Branch A - Mean Pooling → μ:**
     ```python
     # Masked mean pooling
     masked_semantics = semantic_seq * pool_mask  # (B, L, 64)
     sum_semantics = masked_semantics.sum(dim=1)  # (B, 64)
     valid_counts = pool_mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
     mu_raw = sum_semantics / valid_counts  # (B, 64)

     # L2 Normalize AFTER pooling
     mu = F.normalize(mu_raw, p=2, dim=1)  # (B, 64), ||μ|| = 1
     ```

     **Branch B - Max Pooling → κ (专注度是强度量):**
     ```python
     # Masked max pooling - 专注度是强度量，与句子长度无关
     masked_energies = energy_seq * pool_mask  # (B, L, 1)
     # 将 padding 位置的 energy 设为 -inf，确保 max 不选到它们
     masked_energies = masked_energies.masked_fill(pool_mask == 0, float('-inf'))
     mass = masked_energies.max(dim=1).values  # (B, 1)

     # Physical scaling
     kappa = 1.0 + self.alpha_scale * mass  # (B, 1)
     ```

  5. **Branch C (Auxiliary):**
     ```python
     # Connect to pooled mu for supervision
     aux_logits = self.aux_head(mu)  # (B, 28)
     ```

  6. **Return:**
     ```python
     return {
         "mu": mu,              # (B, 64) - semantic direction (entity or sentence)
         "kappa": kappa,        # (B, 1)  - concentration parameter
         "mass": mass,          # (B, 1)  - raw mass (for visualization)
         "aux_logits": aux_logits  # (B, 28) - for L_Aux
     }
     ```

**V4 关键设计变更:**

| 变更点 | V3 (旧) | V4 (新) |
|--------|---------|---------|
| 投影位置 | 先取 [CLS] token，再投影 | 先投影整个序列，再 Pooling |
| Branch A 输入 | `cls_emb` (B, 768) | `last_hidden` (B, L, 768) |
| Branch A 输出 | `mu` (B, 64) 直接 | `semantic_seq` (B, L, 64) → Pooling |
| Branch B 机制 | Gravitational Attention (双向) | **Max Pooling** (专注度是强度量) |
| Pooling 方式 | 无 (直接用 [CLS]) | Masked Mean (A) / **Max** (B) |
| 实体支持 | ❌ 仅支持整句 | ✅ 支持 Token 粒度实体 |

**Masking 示例:**

```python
# 整句模式 (entity_mask=None)
text = "I am absolutely furious right now!"
# attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] (全 1)
# pool_mask:     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] (全句)

# 实体模式 (entity_mask 提供)
text = "The movie was fantastic but acting terrible"
# entity_mask:   [0, 0, 0, 1, 1, 1, 1, 0, 0, 0] (仅 "fantastic")
# pool_mask:     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0] (仅对实体 pooling)
```

**Branch C 监督作用:** Branch C 依然接在 **Pooling 后的 mu** 上，对最终提取的实体/句子向量进行语义监督。如果 Pooling 丢失了关键信息，Branch C 的 KL 损失会惩罚模型。

### 4.4 `src/loss.py` (损失函数) - **Supervised vMF-NCE**

实现 Supervised vMF-NCE 对比学习 + Calibration + Auxiliary Loss + **实体级监督**。

**核心变更：放弃 SimCSE，采用监督信号**

SimCSE 的问题：将同类情感样本（如 "I love cats" 和 "I adore dogs"）视为负例，强迫模型学习"句子相似度"而非"情感相似度"。

**Solution:** Class-Prototype Supervised vMF-NCE - 拉近样本与其情感类别中心的距离。

* **类:** `GBERTLoss(nn.Module)`
* **`__init__(config, num_emotions=28, embedding_dim=64)`:**
  * 保存权重 `lambda_cal`, `lambda_aux`
  * **创建 Class Prototypes:** `self.prototypes = nn.Parameter(torch.randn(num_emotions, embedding_dim))`
    * 可学习参数：28 个情感类别的中心向量
    * 初始化为 L2 归一化的随机向量

* **`forward(outputs, soft_labels)`:**

  **Single Forward (不再需要双前向):**
  - `outputs`: 单次前向传播的输出
  - `soft_labels`: (B, 28) Soft Label 分布

  1. **Extract:** 获取 `mu`, `kappa`, `aux_logits`。

  2. **Calc $L_{vMF}$ (Supervised Class-Prototype 主损失):**
     ```python
     # Step A: 归一化 Prototypes (确保在超球面上)
     prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)

     # Step B: 计算 Cosine Similarity
     # mu: (B, 64), prototypes_norm: (28, 64) -> logits: (B, 28)
     logits = torch.matmul(mu, prototypes_norm.T)

     # Step C: 动态温度 (关键：Detach Kappa 防止梯度回传!)
     # L_vMF 只更新 mu 和 prototypes，kappa 由 L_Cal 单独优化
     tau = 1.0 / (kappa.detach() + 1e-6)  # (B, 1)
     scaled_logits = logits / tau  # (B, 28)

     # Step D: Soft Label 作为目标分布 (多标签支持)
     log_probs = F.log_softmax(scaled_logits, dim=1)  # (B, 28)
     L_vMF = F.kl_div(log_probs, soft_labels, reduction='batchmean')
     ```

  3. **Calc $L_{Cal}$ (校准损失):**
     * **关键修正：** 排除 Neutral 类别（index=27）计算 Max-Norm，防止中性句子获得高κ
     * 计算 Target: `kappa_tgt = 1.0 + 50.0 * torch.max(soft_labels[:, :27], dim=1).values`
     * 计算 MSE Loss: `F.mse_loss(kappa.squeeze(), kappa_tgt)`

  4. **Calc $L_{Aux}$ (辅助损失):**
     * 计算 KL Divergence: `F.kl_div(F.log_softmax(aux_logits, dim=1), soft_labels, reduction='batchmean')`

  5. **Sum:** `L_total = L_vMF + lambda_cal * L_Cal + lambda_aux * L_Aux`

**关键差异对比:**

| 特性 | SimCSE (旧) | Supervised vMF-NCE (新) |
|------|-------------|-------------------------|
| 正对定义 | 同一输入的 Dropout 变体 | 同情感类别的所有样本 |
| 负对定义 | Batch 内其他所有样本 | 不同情感类别的中心 |
| 前向次数 | 2 次 (同一输入) | 1 次 |
| Label 使用 | 仅用于 Calibration | 直接指导 Embedding 学习 |
| 收敛速度 | 慢 (需要大量样本) | 快 (有明确的类别目标) |

**关键设计细节:**

| 设计点 | 实现 | 目的 |
|--------|------|------|
| **梯度截断** | `kappa.detach()` | 确保 $\kappa$ 仅由 $L_{Cal}$ 优化，防止"作弊" |
| **原型归一化** | `F.normalize(self.prototypes)` | 确保原型在超球面上，防止模长膨胀 |

**热力学解释 (保留):** 动态温度 τ 使系统在不同"热力学态"间自适应切换：
* **Solid State (κ → 50, τ → 0):** 低温低熵，高精度检索
* **Gaseous State (κ → 1~10, τ 升高):** 高温高熵，高泛化性

### 4.5 `src/utils.py` (工具函数)

* **`Logger`:** 统一的日志接口
  * 默认输出到 Console 和文件 `logs/train_{timestamp}.log`
  * 检测 WandB: 如果 `wandb` 已安装且登录，自动初始化

* **`set_seed(seed)`:** 设置随机种子 (torch, numpy, random)

* **`get_device()`:** 动态获取设备，支持 CUDA/MPS/CPU

### 4.6 `train.py` (训练循环)

主控制流，包含硬件适配、学习率调度、早停等逻辑。

* **功能流程:**

  1. **硬件检测:**
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if torch.cuda.is_available():
      torch.backends.cudnn.benchmark = True
      logger.info(f"🚀 GPU Activated: {torch.cuda.get_device_name(0)}")
  else:
      logger.warning("⚠️ GPU Not Found. Falling back to CPU. Training will be slow.")
  ```

  2. 初始化 `Config`, `Tokenizer`, `Model`, `Loss`, `Optimizer`, `Scheduler`。
  3. **LR Scheduler:** `get_linear_schedule_with_warmup`
     - `warmup_steps = total_steps * warmup_ratio`
  4. 模型 `.to(device)`。
  5. **Early Stopping 变量:** `best_val_loss = inf`, `patience_counter = 0`。
  6. **Loop Epochs:**
     * **Train Mode:**
       * **Loop Batches:**
         * **Single Forward (Supervised vMF-NCE):** `outputs = model(batch)` (不再需要双前向)
         * 计算 Loss: `loss = criterion(outputs, batch['soft_label'])`
         * **Gradient Accumulation:**
           ```python
           loss = loss / config.grad_accum_steps
           loss.backward()
           if (step + 1) % config.grad_accum_steps == 0:
               optimizer.step()
               scheduler.step()
               optimizer.zero_grad()
           ```
         * Logging (每 N 步): loss components, learning rate, avg kappa
     * **Val Mode (每 Epoch 结束):**
       * 计算 Val Loss (不更新梯度)
       * 记录 Val_Total_Loss, Val_vMF_Loss, Val_Cal_Loss
       * 打印一个 Batch 的 Average Kappa (监控语义塌缩)
     * **Checkpoint & Early Stopping:**
       ```python
       if val_loss < best_val_loss:
           best_val_loss = val_loss
           patience_counter = 0
           torch.save(model.state_dict(), "checkpoints/best_model.pt")
       else:
           patience_counter += 1
       torch.save({
           'model': model.state_dict(),
           'optimizer': optimizer.state_dict(),
           'scheduler': scheduler.state_dict(),
           'epoch': epoch,
           'best_val_loss': best_val_loss,
       }, "checkpoints/last_checkpoint.pt")
       if patience_counter >= config.patience:
           logger.info("Early stopping triggered")
           break
       ```

### 4.7 `inference.py` (交互式推理 Demo) - **V4: Entity-Aware**

**功能:** 单句情感分析 + 实体级情感提取工具（支持 span_text 输入）

* **类:** `GbertPredictor`
  * `load_model(checkpoint_path)`: 加载模型权重
  * `predict(text, span_text=None)`: 返回预测结果
    * `span_text=None`: 整句模式
    * `span_text="fantastic"`: 实体模式（自动字符对齐）

* **模式 1: 整句模式 (entity_mask=None)**
  ```python
  >>> predictor = GbertPredictor.load("checkpoints/best_model.pt")
  >>> result = predictor.predict("I am absolutely furious right now!")
  >>> print(result)
  {
      "text": "I am absolutely furious right now!",
      "mode": "sentence",
      "category": "anger",           # Soft Label Top-1
      "intensity": 0.85,              # I_raw = max(soft_label)
      "kappa": 43.2,                  # 物理质量
      "mu": [0.12, -0.34, ...]        # 64d 向量 (显示前5维)
  }
  ```

* **模式 2: 实体模式 (entity_mask 指定)** - V4 新增
  ```python
  >>> text = "The movie was fantastic but acting terrible"
  >>> # 对 "fantastic" 做情感分析
  >>> result = predictor.predict(
  ...     text,
  ...     span_text="fantastic"  # 使用原始文本，而非 token 索引
  ... )
  >>> print(result)
  {
      "text": "The movie was fantastic but acting terrible",
      "mode": "entity",
      "span_text": "fantastic",
      "category": "joy",
      "intensity": 0.72,
      "kappa": 37.1,
      "mu": [0.08, 0.21, ...]        # 仅基于 "fantastic" 的语义向量
  }
  ```

* **内部实现 (自动字符对齐 + Empty Mask 保护):**
  ```python
  def predict(self, text, span_text=None):
      """
      Args:
          text: 输入文本
          span_text: 实体文本 (如 "fantastic")，如果为 None 则分析整句
      """
      encoding = self.tokenizer(text, return_offsets_mapping=True)

      if span_text is None:
          entity_mask = None  # 整句模式
      else:
          # 使用与 Dataset 相同的字符对齐逻辑
          entity_mask = self._create_entity_mask(text, span_text, encoding)

      inputs = {
          'input_ids': torch.tensor(encoding['input_ids']).unsqueeze(0),
          'attention_mask': torch.tensor(encoding['attention_mask']).unsqueeze(0),
          'entity_mask': torch.tensor(entity_mask).unsqueeze(0) if entity_mask is not None else None
      }

      # 关键修正（Empty Mask 保护）：检查 entity_mask 是否全为 0
      if entity_mask is not None and entity_mask.sum() == 0:
          import warnings
          warnings.warn(f"Entity '{span_text}' not found in text. Falling back to sentence-level analysis.")
          inputs['entity_mask'] = None

      with torch.no_grad():
          outputs = self.model(**inputs)

      return self._format_result(text, span_text, outputs)
  ```

---

## 5. 关键实现细节 (Key Implementation Details)

### 5.1 Token-Level Pooling (V4 核心)

V4 架构的核心创新是将投影层移到 Backbone 之后、Pooling 之前：

```python
# V3 (旧): [CLS] Token 直接投影
cls_emb = last_hidden[:, 0, :]  # (B, 768)
mu = F.normalize(self.semantic_head(cls_emb), dim=1)

# V4 (新): 先投影整个序列，再 Pooling
semantic_seq = self.semantic_head(last_hidden)  # (B, L, 64)
# Masked Mean Pooling
masked = semantic_seq * pool_mask.unsqueeze(-1)  # (B, L, 64)
mu_raw = masked.sum(dim=1) / pool_mask.sum(dim=1).clamp(min=1e-9)  # (B, 64)
mu = F.normalize(mu_raw, p=2, dim=1)
```

**优势:**
1. **实体感知:** 可以仅对特定 Token 做 Pooling，提取实体语义
2. **更丰富的语义:** 不再依赖 [CLS] 的预设位置，使用实际的语义 Token
3. **灵活的注意力:** 可以通过调整 entity_mask 实现软/硬注意力

### 5.2 Supervised vMF-NCE 训练流程 (对比学习核心)

```python
# 训练时，单次前向传播
for batch in dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}

    # Single Forward - Supervised vMF-NCE
    outputs = model(batch['input_ids'], batch['attention_mask'], batch['entity_mask'])

    # 计算损失 (使用 Class Prototypes)
    loss = criterion(outputs, batch['soft_label'])
```

### 5.3 硬件回退策略 (Hardware Fallback)

在所有涉及 Tensor 运算的地方，都不硬编码 `.cuda()`，而是使用 `config.device` 或 `tensor.to(device)`。

### 5.4 数值稳定性 (Numerical Stability)

* **Kappa 下界:** `tau = 1.0 / (kappa + 1e-6)`
* **Softplus:** `token_energies = F.softplus(self.energy_proj(last_hidden))`

### 5.5 模拟大 Batch (Gradient Accumulation)

```python
loss = loss / config.grad_accum_steps  # 缩放 loss
loss.backward()
if (step + 1) % config.grad_accum_steps == 0:
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```
**Effective Batch Size = Physical × Accum = 64 × 4 = 256**

### 5.7 字符对齐问题详解 (V4 Critical)

**问题根源:**

Tokenizer 的 WordPiece/BPE 算法会将单词拆分为多个 Subword Token：
```python
text = "I absolutely love it!"
tokens = tokenizer.tokenize(text)
# ['I', 'absolutely', 'love', 'it', '!']

# 但如果用户标注的是 "absolutely" 这个词
# 在 Token 序列中的位置是多少？直接用单词索引是错误的！
```

**错误做法 (严禁):**
```python
# ❌ 假设 "absolutely" 是第 2 个单词，就认为 token_idx = 1
# 这在 Subword Tokenization 下完全错误！
```

**正确做法 (offset_mapping):**
```python
# ✅ 使用字符级对齐
encoding = tokenizer(text, return_offsets_mapping=True)
# offset_mapping = [(0, 1), (2, 12), (13, 17), (18, 20), (20, 21)]
#                    'I'  'absolutely'  'love'   'it'     '!'

# 用户标注 "absolutely" (char_start=2, char_end=12)
# 遍历 offset_mapping，找到完全包含在 [2, 12) 区间内的 Token
span_text = "absolutely"
# 数据中已提供: char_start=2, char_end=12 (V4: 显式 offset，无需 text.find())
c_start, c_end = item['char_start'], item['char_end']

# V4: 使用向量化操作计算 entity_mask
token_starts = offsets[:, 0]
token_ends = offsets[:, 1]

# Token 与 Entity 有交集 → True
entity_mask = (token_starts < c_end) & (token_ends > c_start) & attention_mask.bool()
# 结果: [0, 1, 0, 0, 0] - 只有 "absolutely" 被标记
```

**边界情况处理:**

1. **部分重叠:** 使用交集判断 (`<` and `>`)，而非完全包含
2. **重复词问题:** 显式 offset 精确定位，无歧义（V4 关键改进）

**V4 关键改进：** 数据中显式包含 `char_start` 和 `char_end`，不再使用 `text.find()`，避免了重复词时的歧义问题。

**Subword 示例:**
```python
text = "I love transformers"
# Tokenizer: ['I', 'love', 'transform', '##ers']
# offset_mapping: [(0,1), (2,6), (7,16), (16,20)]

# 标注 "transformers"
char_start = 7
char_end = 19
# 对齐结果: [0, 0, 1, 1] - "transform" + "##ers" 两个 Token
```

### 5.8 GoEmotions 数据处理示例 (Legacy)

```python
# 假设原始格式
# {
#   "text": "I'm so happy!",
#   "labels": [0, 0, 5, 1, 0, ...]  # 28 类别，每个是标注人数
# }

# 归一化为 Soft Label
import torch
labels = torch.tensor([0, 0, 5, 1, 0, ...], dtype=torch.float32)
soft_label = labels / labels.sum()  # 归一化
intensity = torch.max(soft_label).item()  # Max-Norm
```

---

## 6. 开发步骤 (Action Plan)

1. **环境准备:** 创建 `requirements.txt` (torch, transformers, scikit-learn, tqdm, wandb[optional])。
2. **数据准备:** 使用 GoEmotions Raw Annotations 生成 `train.jsonl`（归一化投票数）。
3. **核心模块:** 依次实现 `config.py` → `model.py` → `loss.py` → `dataset.py` → `utils.py`。
4. **训练脚本:** 编写 `train.py`，先用 CPU 在少量数据上跑通流程 (Debug mode)。
5. **推理脚本:** 编写 `inference.py` 交互式 Demo。
6. **全量训练:** 切换到 GPU 进行完整训练。

---

## 7. 依赖包 (Requirements)

```
torch>=2.0.0
transformers>=4.30.0
scikit-learn
tqdm
wandb  # Optional, 用于实验跟踪
openai  # Optional, 用于 generate_data.py
datasets  # GoEmotions 数据集加载
```
