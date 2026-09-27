# P0-CNN-Evidence

> **项目**: CNN裂缝检测 (jinliangyue/concrete-crack-detection)
> **目的**: P0 证据补强 - FPR/FNR 计算 + Confusion Matrix 可视化 + Streamlit Demo 验证
> **生成日**: 2026-09-24

## 目录结构

```
P0-Evidence/
├── README.md                  (本文件)
├── metrics/
│   ├── final_metrics.csv      (CSV: 4 个模型 × 12 列指标)
│   └── final_metrics.md       (Markdown: 同上 + 验证报告)
├── confusion_matrix/
│   ├── confusion_matrix.png          (2×2 合并图 4 模型)
│   ├── confusion_matrix_rf.png       (单模型)
│   ├── confusion_matrix_cnn.png
│   ├── confusion_matrix_cnn_se.png
│   └── confusion_matrix_resnet18.png
├── demo/
│   ├── runtime_verification.md       (Streamlit Demo 运行时验证)
│   └── samples/                      (Test A / Test B 推理结果)
└── reproducibility/
    └── environment.md                (环境 + 复现命令)
```

## 核心成果

### P0-1: FPR / FNR

| 模型 | Accuracy | Precision | Recall | F1 | **FPR** | **FNR** | TP | TN | FP | FN |
|------|---------:|----------:|-------:|---:|---:|---:|---:|---:|---:|---:|
| RF | 59.92% | 59.11% | 64.33% | 61.61% | **44.50%** | **35.67%** | 386 | 333 | 267 | 214 |
| CNN | 76.25% | 78.38% | 72.50% | 75.32% | **20.00%** | **27.50%** | 435 | 480 | 120 | 165 |
| CNN+SE | 77.58% | 79.29% | 74.67% | 76.91% | **19.50%** | **25.33%** | 448 | 483 | 117 | 152 |
| ResNet18 | 87.92% | 94.70% | 80.33% | 86.93% | **4.50%** | **19.67%** | 482 | 573 | 27 | 118 |

> 测试集：SDNET2018 · n=1,200 · Crack=600 · NoCrack=600 · 1:1 平衡 · seed=42
> 正类定义：Crack (index 1)
> 来源：`results/<model>_results.json` 中的 `predictions` + `labels` 数组
> 计算函数：`src/confusion.py::confusion_matrix()` + 现有项目内 `per_class_metrics_from_cm()`

### P0-2: Confusion Matrix

- 已生成 5 张 PNG（合并 + 4 个单模型）
- 单元格采用 4 色区分：TN(绿) / TP(绿) / FP(红) / FN(橙)
- 数值与原项目 report 字段完全一致（diff < 0.001）

### P0-3: Streamlit Demo 验证

| 检查项 | 结果 |
|--------|------|
| Python / Streamlit / PyTorch 版本 | ✓ |
| 模型 checkpoint 加载 | ✓ |
| Predict 函数调用 | ✓ 6/6 正确 |
| NoCrack 样本 (3 张) | ✓ 全部预测 NoCrack (97.35% / 99.19% / 71.96%) |
| Crack 样本 (3 张) | ✓ 全部预测 Crack (100.00% / 65.56% / 98.42%) |
| 6 个 Streamlit Section 数据源 | ✓ 全部存在 |
| 运行错误 | 无 |

详细验证见 `demo/runtime_verification.md`

## 数据来源

所有数字均来自真实测试集运行结果（`results/<model>_results.json` 中的 predictions 和 labels 数组），**不是** 推算或假设。

## 与原项目的差异

P0 任务**没有修改任何项目代码**：
- ✓ 模型未变
- ✓ 数据集未变（SDNET2018）
- ✓ Train/Test split 未变
- ✓ Checkpoint 未变
- ✓ Inference 逻辑未变
- ✓ Train 命令未变

**唯一新增**：从已有 predictions + labels 数组计算 TP/TN/FP/FN/FPR/FNR，并生成可视化。

## 验证公式

- Accuracy = (TP + TN) / Total
- Precision(Crack) = TP / (TP + FP)
- Recall(Crack) = TP / (TP + FN)
- F1 = 2 × Precision × Recall / (Precision + Recall)
- FPR = FP / (FP + TN)
- FNR = FN / (FN + TP)

所有公式在 `demo/runtime_verification.md` 中确认与原项目 `per_class_metrics_from_cm()` 完全一致（diff < 0.001）。

## 答辩可用事实

> **团队 CNN 裂缝检测项目** 在 SDNET2018 真实数据集（56,092 张混凝土桥面/路面/墙体图像）上完成模型训练与独立测试集评价。测试集 1,200 张（Crack=600, NoCrack=600, 1:1 平衡, seed=42）。4 个模型对比完整，ResNet18（ImageNet 迁移）实测 accuracy=87.92%，Precision=94.70%，Recall=80.33%，F1=86.93%，**FPR=4.50%**，**FNR=19.67%**。Confusion Matrix 已生成可视化。该实践为标准中 AI 辅助缺陷诊断性能评价方法（A-H 八子节）的设计提供了具体工程实践基础。

## 不能证明的事

> 项目当前状态**没有**实现：
> - 多源异构数据融合标准实现
> - 跨系统数据互操作标准实现
> - 可信数据流转标准实现
> - 多类别缺陷定位（像素级分割）
> - 工程现场部署（非公开数据集）
> - 实时性能（推理时延 ms 级）

---

**生成完毕。P0 任务停止。**