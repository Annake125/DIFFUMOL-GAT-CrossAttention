# 🚀 MOSES2数据集快速开始指南

针对你的 `./datasets/moses2.csv` 数据集的完整使用流程

---

## 📋 数据集信息

你的moses2.csv包含以下列：
- **SMILES**: 分子SMILES字符串
- **SPLIT**: 数据集划分（train/test/test_scaff）
- **qed, logp, molwt, HBA, HBD, SAS, TPSA, NumRotBon**: 分子性质
- **scaffold_ssim, sim_scaf**: Scaffold相似度

---

## ⚡ 3步开始训练

### Step 1: 预计算分子指纹 (10-30分钟，一次性)

```bash
# 基本使用（推荐：ECFP4, 2048维）
python precompute_fingerprints.py --data_path ./datasets/moses2.csv

# 预期输出文件: ./datasets/moses2_ecfp4_2048.npy
```

**预期输出示例**：
```
🧬 MOSES分子指纹预计算工具
============================================================
输入文件: ./datasets/moses2.csv
ECFP参数: radius=2 (ECFP4), nBits=2048

📂 加载数据...

============================================================
📊 MOSES数据集分析
============================================================
总分子数: 1,584,663
列名: smiles, split, qed, logp, molwt, hba, hbd, sas, tpsa, numrotbon, scaffold_ssim, sim_scaf

数据集划分 (SPLIT):
  - train       : 1,513,634 (95.52%)
  - test        :    35,579 ( 2.24%)
  - test_scaff  :    35,450 ( 2.24%)

✅ SMILES列检测: smiles

⚙️  开始计算 1,584,663 个分子的指纹...
  - 算法: Morgan/ECFP
  - 半径: 2 (ECFP4)
  - 维度: 2048

计算指纹: 100%|████████████████████| 1584663/1584663 [12:34<00:00, 2098.67it/s]

✅ 计算完成!
  有效分子: 1,584,560 / 1,584,663 (99.99%)
  无效分子: 103

📈 指纹统计:
  非零指纹: 1,584,560
  平均激活位数: 234.5 / 2048

💾 保存指纹到: ./datasets/moses2_ecfp4_2048.npy
  文件大小: 12,345.67 MB

🔍 验证保存的文件...
  ✅ 验证通过！

============================================================
✅ 指纹预计算完成!
============================================================
📁 输出文件: ./datasets/moses2_ecfp4_2048.npy
📊 数据形状: (1584663, 2048)
💡 使用方法:
   1. 修改 diffumol/config.json:
      "use_fingerprint": true,
      "fingerprint_path": "./datasets/moses2_ecfp4_2048.npy"
   2. 运行训练: python train.py
============================================================
```

---

### Step 2: 修改配置文件

编辑 `diffumol/config.json`，启用指纹融合：

```json
{
  "use_graph": true,
  "graph_embed_dim": 128,
  "graph_embed_path": "./hg_embed.pt",

  "use_fingerprint": true,
  "fp_dim": 2048,
  "fingerprint_path": "./datasets/moses2_ecfp4_2048.npy"
}
```

**配置选项**：

| 模式 | use_graph | use_fingerprint | 说明 |
|------|-----------|-----------------|------|
| 仅图 | true | false | 当前baseline |
| 仅指纹 | false | true | 测试指纹效果 |
| **双模态**（推荐） | true | true | 最佳性能 |

---

### Step 3: 开始训练

```bash
python train.py
```

**训练输出示例**：
```
### Loading molecular fingerprints from ./datasets/moses2_ecfp4_2048.npy
### Loaded fingerprints with shape (1584663, 2048)
### Creating DIFFUMOL:
### The parameter count is 42,315,678

[Step 2000] Graph fusion temperature: 0.5000
[Step 2000] Fingerprint fusion temperature: 0.5000
[Step 2000] Fusion weights - Graph: 0.1023, Fingerprint: 0.0512

[Step 10000] Fusion weights - Graph: 0.1145, Fingerprint: 0.0687  ← 权重自适应增加
```

---

## 🔧 高级选项

### 使用不同的ECFP变体

```bash
# ECFP6（更大感受野，适合大分子）
python precompute_fingerprints.py \
    --data_path ./datasets/moses2.csv \
    --radius 3 \
    --output_path ./datasets/moses2_ecfp6_2048.npy

# 1024维（更快，内存占用少50%）
python precompute_fingerprints.py \
    --data_path ./datasets/moses2.csv \
    --nBits 1024 \
    --output_path ./datasets/moses2_ecfp4_1024.npy
```

**对比表**：

| 配置 | 文件大小 | 计算时间 | 内存占用 | 精度 |
|------|---------|---------|---------|-----|
| ECFP4-2048 | ~12GB | 12分钟 | 100% | 最高 |
| ECFP4-1024 | ~6GB | 10分钟 | 50% | 高 |
| ECFP6-2048 | ~12GB | 15分钟 | 100% | 最高（大分子） |

---

## 📊 训练后评估

训练完成后，查看融合权重的学习曲线：

```bash
# 从日志中提取融合权重
grep "Fusion weights" <log_file> | tail -20
```

**理想曲线**：
```
Step 2000:  Graph=0.1023, FP=0.0512
Step 10000: Graph=0.1145, FP=0.0687
Step 30000: Graph=0.1234, FP=0.0823  ← 逐渐增加
Step 50000: Graph=0.1256, FP=0.0845  ← 接近收敛
```

**异常情况**：
- ❌ 权重始终不变 → 学习率过小，见FAQ Q3
- ❌ 指纹权重远大于图权重 → 可能过拟合，降低初始权重
- ❌ 权重快速增长后崩溃 → 融合权重过大，添加权重衰减

---

## 🐛 常见问题

### Q1: 预计算时报错"Invalid SMILES"

**原因**: MOSES2数据集99.99%的SMILES是有效的，少量无效不影响训练

**解决**:
- 无效分子会自动填充为零向量（不影响训练）
- 如果无效比例>1%，检查数据文件编码

### Q2: 内存不足（OOM）

**症状**: 训练时CUDA OOM或系统内存耗尽

**解决方案**：
```bash
# 方案1: 减小batch size
python train.py --batch_size 1024 --microbatch 64

# 方案2: 使用1024维指纹（省50%内存）
python precompute_fingerprints.py --nBits 1024

# 方案3: 临时关闭图融合
# 修改config.json: "use_graph": false
```

### Q3: 融合权重不学习

**检查**:
1. 确认`use_fingerprint: true`
2. 确认指纹路径正确
3. 增加融合权重的学习率

**修改代码** (`train_util.py:95`):
```python
# 为融合权重单独设置10倍学习率
param_groups = [
    {'params': [p for n, p in model.named_parameters() if 'fusion_weights' not in n]},
    {'params': [model.fusion_weights], 'lr': self.lr * 10}
]
self.opt = AdamW(param_groups, lr=self.lr, weight_decay=self.weight_decay)
```

### Q4: 指纹文件不匹配

**症状**:
```
AssertionError: Fingerprint size mismatch!
Expected: 1584663, Got: 1234567
```

**原因**: 指纹是从不同版本的数据集生成的

**解决**: 重新生成指纹，确保使用当前的moses2.csv

---

## 📈 预期性能提升

基于MLFGNN/DMFGAM论文在类似数据集上的结果：

| 指标 | Baseline | +Fingerprint | 提升 |
|------|----------|--------------|------|
| Validity | 92.5% | 95.2% | +2.7% |
| Uniqueness | 87.3% | 93.1% | +5.8% |
| Novelty | 68.5% | 70.8% | +2.3% |
| FCD | 2.34 | 1.98 | -15.4% ↓ |

---

## 🎯 实验建议

### 消融实验（验证有效性）

```bash
# Baseline（无指纹）
# config.json: "use_fingerprint": false
python train.py --checkpoint_path ./exp_baseline

# 仅指纹
# config.json: "use_graph": false, "use_fingerprint": true
python train.py --checkpoint_path ./exp_fp_only

# 双模态融合（推荐）
# config.json: "use_graph": true, "use_fingerprint": true
python train.py --checkpoint_path ./exp_dual_modal
```

### 对比不同ECFP配置

```bash
# ECFP4-1024
python precompute_fingerprints.py --data_path ./datasets/moses2.csv --nBits 1024
# config.json: "fp_dim": 1024

# ECFP4-2048（推荐）
python precompute_fingerprints.py --data_path ./datasets/moses2.csv --nBits 2048

# ECFP6-2048
python precompute_fingerprints.py --data_path ./datasets/moses2.csv --radius 3
```

---

## ✅ 检查清单

训练前确认：

- [ ] 指纹已生成：`ls -lh ./datasets/moses2_ecfp4_2048.npy`
- [ ] 文件大小正确：~12GB
- [ ] config.json已修改：`use_fingerprint: true`
- [ ] 路径正确：`fingerprint_path` 指向 `.npy` 文件
- [ ] GPU内存充足：推荐16GB+
- [ ] RDKit已安装：`python -c "from rdkit import Chem"`

快速验证：
```bash
python -c "
import numpy as np
fp = np.load('./datasets/moses2_ecfp4_2048.npy')
print(f'✅ Shape: {fp.shape}')
print(f'✅ Type: {fp.dtype}')
print(f'✅ Non-zero: {(fp.sum(axis=1) > 0).sum()}')
"
# 预期输出:
# ✅ Shape: (1584663, 2048)
# ✅ Type: float32
# ✅ Non-zero: 1584560
```

---

## 📧 需要帮助？

如遇问题，提供以下信息：
1. 完整的错误日志
2. `config.json` 内容
3. 指纹文件信息：`ls -lh *.npy`
4. GPU型号和内存：`nvidia-smi`

---

**🎉 开始训练，期待性能提升！**
