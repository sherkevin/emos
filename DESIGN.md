这是一个为 **Probabilistic G-BERT (System V3)** 量身定制的工程实现设计文档。

这份文档将指导你（或 Claude Code）构建整个项目。它采用了模块化设计，确保代码结构清晰、易于调试，并且内置了 **自动硬件检测 (GPU/CPU)** 逻辑。

---

# Probabilistic G-BERT: 工程实现设计文档 (Implementation Design Doc)

## 1. 项目概览 (Project Overview)

* **项目名称:** `probabilistic_gbert`
* **核心架构:** Bottlenecked Tri-Branch (RoBERTa → 64d Semantic μ → Mass κ + Aux logits)
* **架构流向:** **串行结构** - Branch C 接在 Branch A 输出端，形成信息瓶颈约束
* **训练目标:** 3-Part Loss (vMF-NCE + Calibration + Auxiliary)
* **硬件策略:** 优先使用 CUDA (GPU)，若不可用自动回退至 CPU。
* **开发框架:** Python 3.9+, PyTorch 2.0+, Transformers

---

## 2. 项目目录结构 (Directory Structure)

```text
probabilistic_gbert/
├── data/                        # 数据存储
│   ├── raw/                     # 原始数据
│   ├── processed/               # 处理后的 JSONL (带 Soft Labels)
│   └── generate_data.py         # LLM 数据生成脚本 (需包含 PRD 中的 Soft Label Prompt)
├── src/                         # 源代码
│   ├── __init__.py
│   ├── config.py                # 全局配置参数 (Hyperparameters)
│   ├── dataset.py               # 数据加载与处理 (Soft Label Max-Norm)
│   ├── model.py                 # PyTorch 模型定义 (Tri-Branch 串行结构)
│   ├── loss.py                  # 损失函数定义 (MATS Loss)
│   └── utils.py                 # 工具函数 (Logger, Metrics, Device)
├── checkpoints/                 # 模型保存路径
├── train.py                     # 训练主入口
├── inference.py                 # 推理与检索测试脚本
├── requirements.txt             # 依赖包
└── README.md                    # 项目说明
```

---

## 3. 模块功能详细描述 (Module Specifications)

### 3.1 `src/config.py` (配置中心)

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
| `effective_batch_size` | 256 | 对比学习目标 Batch 大小 |
| `grad_accum_steps` | 4 | 梯度累积步数 (Physical × Accum = Effective) |
| `lr_backbone` | 2e-5 | Backbone 学习率 |
| `lr_heads` | 1e-4 | 分支学习率 |
| `epochs` | 5 | 训练轮数 |
| `lambda_cal` | 0.1 | 校准损失权重 |
| `lambda_aux` | 0.05 | 辅助损失权重 |
| `device` | 动态获取 | "cuda" or "cpu" |

### 3.2 `src/dataset.py` (数据管道)

负责加载 JSONL 数据并计算强度。

* **类:** `EmotionDataset(Dataset)`
* **`__init__(data_path, tokenizer, max_len)`:** 加载数据。
* **`__getitem__(idx)`:**
  1. Tokenize 文本。
  2. 加载 Soft Label 分布 (List -> FloatTensor)。
  3. **核心逻辑:** 实时计算强度 `intensity = torch.max(soft_label)`。
  4. 返回字典: `input_ids`, `attention_mask`, `soft_label`, `intensity`。

* **函数:** `create_dataloaders(config)`
  * 返回 Train/Val DataLoader。

### 3.3 `src/model.py` (模型架构)

实现 System V3 的三分支**串行**结构。

**架构流向:**
```
RoBERTa (768d)
    ├─→ Branch A (Semantic) → μ (64d) ─→ Branch C (Auxiliary) → logits (28d)
    └─→ Branch B (Mass)     → κ (1d)
```

* **类:** `ProbabilisticGBERT(nn.Module)`

* **`__init__(config)`:**
  * 初始化 Backbone (RoBERTa)。
  * **Branch A:** `Sequential(Linear(768→256), GELU, Linear(256→64))`。
  * **Branch B:** `EnergyProj(768→1)`, `AttnProj(768→1)`。
  * **Branch C:** `Sequential(Linear(64→128), GELU, Linear(128→28))`。

* **`forward(input_ids, mask)`:**
  * 提取 `[CLS]` token (768d) 和 `last_hidden_state` (B×L×768)。
  * **Branch A:** 计算 `mu` (64d)，**必须做** `F.normalize(mu, p=2, dim=1)`。
  * **Branch B:**
    * 计算注意力权重和能量
    * 应用公式: `kappa = 1.0 + alpha_scale * mass`
  * **Branch C:** **以 Branch A 的输出 `mu` 为输入**，计算 `aux_logits` (28d)。
  * **Return:** 包含 `mu`, `kappa`, `aux_logits`, `mass` 的字典。

**关键设计思想:** Branch C 接在 Branch A 之后，对 64d 瓶颈向量进行解码。如果 Branch A 丢失了语义信息，Branch C 的 KL 散度损失会爆炸，从而惩罚 Branch A。这形成了**信息瓶颈约束**。

### 3.4 `src/loss.py` (损失函数)

实现 NIPS 核心数学公式。

* **类:** `GBERTLoss(nn.Module)`
* **`__init__(config)`:** 保存权重 `lambda_cal`, `lambda_aux`。

* **`forward(outputs, soft_labels)`:**

  1. **Extract:** 从 `outputs` 字典获取 `mu`, `kappa`, `aux_logits`。

  2. **Calc $L_{vMF}$ (主损失):**
     * 计算 Cosine Similarity Matrix: `logits = torch.matmul(mu, mu.T)`
     * 计算动态温度: `tau = 1.0 / (kappa + 1e-6)`
     * 缩放 Logits: `scaled_logits = logits / tau`
     * 计算 CrossEntropy (Label 为 `torch.arange(B)`, 假设每个样本的下个样本是其正对)

  3. **Calc $L_{Cal}$ (校准损失):**
     * 计算 Target: `kappa_tgt = 1.0 + 50.0 * torch.max(soft_labels, dim=1).values`
     * 计算 MSE Loss: `F.mse_loss(kappa.squeeze(), kappa_tgt)`

  4. **Calc $L_{Aux}$ (辅助损失):**
     * 计算 KL Divergence: `F.kl_div(F.log_softmax(aux_logits, dim=1), soft_labels, reduction='batchmean')`

  5. **Sum:** `L_total = L_vMF + lambda_cal * L_Cal + lambda_aux * L_Aux`

**热力学解释:** 动态温度 τ 使系统在不同"热力学态"间自适应切换：
* **Solid State (κ → 50, τ → 0):** 低温低熵，高精度检索
* **Gaseous State (κ → 1~10, τ 升高):** 高温高熵，高泛化性

### 3.5 `train.py` (训练循环)

主控制流，包含硬件适配逻辑。

* **功能流程:**

  1. **硬件检测:**
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if torch.cuda.is_available():
      torch.backends.cudnn.benchmark = True
      print(f"🚀 GPU Activated: {torch.cuda.get_device_name(0)}")
  else:
      print("⚠️ GPU Not Found. Falling back to CPU. Training will be slow.")
  ```

  2. 初始化 `Config`, `Tokenizer`, `Model`, `Loss`, `Optimizer`。
  3. 模型 `.to(device)`。
  4. 初始化 `Optimizer` (分组学习率：Backbone 用小 LR，Heads 用大 LR)。
  5. **Loop Epochs:**
     * **Loop Batches:**
       * 数据 `.to(device)`。
       * Forward Pass。
       * Compute Loss。
       * **Gradient Accumulation:**
         ```python
         loss = loss / config.grad_accum_steps
         loss.backward()
         if (step + 1) % config.grad_accum_steps == 0:
             optimizer.step()
             optimizer.zero_grad()
         ```
       * Logging (Console / WandB)。
  6. **Save Model:** 保存 `state_dict`。

---

## 4. 关键实现细节 (Key Implementation Details)

### 4.1 硬件回退策略 (Hardware Fallback)

在所有涉及 Tensor 运算的地方，都不硬编码 `.cuda()`，而是使用 `config.device` 或 `tensor.to(device)`。

```python
# 示例：train.py
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    device_name = torch.cuda.get_device_name(0)
    print(f"🚀 GPU Activated: {device_name}")
else:
    print("⚠️ GPU Not Found. Falling back to CPU. Training will be slow.")
```

### 4.2 数值稳定性 (Numerical Stability)

* **Kappa 下界:** 在计算 `tau = 1/kappa` 时，防止 `kappa` 过小导致除零。
  * 代码: `tau = 1.0 / (kappa + 1e-6)`

* **Softplus:** 计算能量时使用 `F.softplus()` 保证质量非负。
  * 代码: `token_energies = F.softplus(self.energy_proj(last_hidden))`

### 4.3 模拟大 Batch (Contrastive Requirement)

对比学习依赖 Batch 内负样本数量。

* 如果使用 GPU (如 16GB VRAM)，Physical Batch Size 可能只能开到 64。
* **必须实现** 梯度累积，逻辑如下：
```python
loss = loss / config.grad_accum_steps  # 缩放 loss
loss.backward()
if (step + 1) % config.grad_accum_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```
* 这使得 **Effective Batch Size = Physical × Accum = 64 × 4 = 256**

---

## 5. 开发步骤 (Action Plan)

1. **环境准备:** 创建 `requirements.txt` (torch, transformers, scikit-learn, tqdm)。
2. **数据生成:** 编写 `data/generate_data.py`，**必须使用 PRD 文档中指定的 Soft Label Prompt**，调用 API 准备好 `train.jsonl`。
3. **核心模块:** 依次实现 `config.py` → `model.py` → `loss.py`。
4. **训练脚本:** 编写 `train.py` 并先用 CPU 在少量数据上跑通流程 (Debug mode)。
5. **全量训练:** 切换到 GPU 进行完整训练。
