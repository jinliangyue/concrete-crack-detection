# Final AI Test Results (ResNet18, CNN, CNN+SE, RF)

**Test set**: SDNET2018, n=1,200, balanced (600 Crack / 600 NoCrack)
**Positive class**: Crack (index 1)
**Random seed**: 42
**Source**: results/<model>_results.json (predictions + labels arrays)

## Final Metrics

| Model | TP | TN | FP | FN | Total | Accuracy | Precision | Recall | F1 | FPR | FNR |
|-------|---:|---:|---:|---:|-----:|---------:|----------:|---------:|---:|------:|------:|
| RF | 386 | 333 | 267 | 214 | 1200 | 59.92% | 59.11% | 64.33% | 61.61% | 44.50% | 35.67% |
| CNN | 435 | 480 | 120 | 165 | 1200 | 76.25% | 78.38% | 72.50% | 75.32% | 20.00% | 27.50% |
| CNN_SE | 448 | 483 | 117 | 152 | 1200 | 77.58% | 79.29% | 74.67% | 76.91% | 19.50% | 25.33% |
| RESNET18 | 482 | 573 | 27 | 118 | 1200 | 87.92% | 94.70% | 80.33% | 86.93% | 4.50% | 19.67% |

## Verification

Computed values vs. reported values:

| Model | Accuracy Diff | Precision Diff | Recall Diff | F1 Diff |
|---|---|---|---|---|
| RF | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| CNN | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| CNN_SE | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| RESNET18 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

All differences < 0.001 (rounding tolerance only).

## Definitions

- TP = true Crack, predicted Crack
- TN = true NoCrack, predicted NoCrack
- FP = true NoCrack, predicted Crack (false alarm)
- FN = true Crack, predicted NoCrack (missed detection)
- FPR = FP / (FP + TN) — false alarm rate
- FNR = FN / (FN + TP) — missed detection rate
- Precision (Crack) = TP / (TP + FP)
- Recall (Crack) = TP / (TP + FN)
- F1 (Crack) = 2 × Precision × Recall / (Precision + Recall)
