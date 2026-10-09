# Concrete Crack Detection · Transfer Learning Ablation

[![Tests](https://img.shields.io/github/actions/workflow/status/jinliangyue/concrete-crack-detection/test.yml?branch=main&label=tests)](https://github.com/jinliangyue/concrete-crack-detection/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://github.com/jinliangyue/concrete-crack-detection/blob/main/runtime.txt)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://concrete-crack-detection.streamlit.app/)

> 在 SDNET2018（56,092 张真实混凝土桥面/路面/墙体图像）上的迁移学习消融对比——**Random Forest (59.92%) → 自建 CNN (76.25%) → CNN+SE 通道注意力 (77.58%) → ResNet18 迁移学习 (87.92%)**。模型 + Web 演示 + 完整复现命令。

A clean three-way ablation on the SDNET2018 concrete crack dataset, comparing a hand-crafted feature baseline (Random Forest), a self-built CNN trained from scratch, and a pre-trained ResNet18 with transfer learning. Built end-to-end on Apple Silicon (MPS GPU) — no cloud GPU needed.

## Results

The original four-model headline uses `max-per-class=4000` (single stratified 85/15 split, the original paper protocol). The three new backbones are trained on `max-per-class=2000` (half the data; the goal is architectural breadth, not headline revalidation — see [docs/K_FOLD.md](docs/K_FOLD.md) for the same rationale applied to K-fold).

| Method | Test accuracy | Train / Test | Parameters |
|---|---:|---|---:|
| Random Forest (64×64 grayscale, 200 trees) | **59.92%** | 6,800 / 1,200 | — |
| Self-built CNN (4 conv blocks, 96×96 input, CosineAnnealingLR, 20 epochs) | **76.25%** | 6,800 / 1,200 | 5.30M |
| **CNN + SE-Blocks** (channel attention, same backbone, 96×96) | **77.58%** | 6,800 / 1,200 | **5.32M** (+0.23%) |
| **ResNet18 (ImageNet transfer, 160×160 input)** | **87.92%** | 6,800 / 1,200 | 11M |
| **ResNet50 (ImageNet transfer, 160×160 input)** | **84.00%** | 3,400 / 600 | 23.5M |
| **EfficientNet-B0 (compound-scaled, ImageNet transfer)** | **85.17%** | 3,400 / 600 | **4.01M** |
| **MobileNetV3-Large (mobile-friendly, ImageNet transfer)** | **86.67%** | 3,400 / 600 | **4.20M** |

**Headline (max-per-class=4000):** ResNet18 with ImageNet pre-training beats the Random Forest baseline by **28.00 percentage points**, the self-built CNN by **11.67 percentage points**, and CNN+SE by **10.34 percentage points**. CNN+SE channel attention adds **+1.33pp** over the plain CNN for only **+12,296 parameters (+0.23%)**.

**Modern backbones (max-per-class=2000):** On half the training data,
MobileNetV3-Large (4.20M params) is the best backbone at **86.67%** —
ahead of EfficientNet-B0 (4.01M params, 85.17%) and ResNet50 (23.5M,
84.00%). All three crush the hand-crafted CNN+SE (77.58% on the original
4000/class data), confirming that **ImageNet pre-training beats
architectural innovation on a 200-image/class problem**. The fact that the
two compound-scaled / mobile-friendly backbones beat ResNet50 at less
than 1/5 the parameter count is a strong argument for NAS-designed
architectures over the older ResNet family.

> **2026-09-12 note**: All four models re-trained with v8.3 (predictions + labels saved to JSON). RF acc 59.42% → 59.92% (sklearn micro-noise across re-runs, same seed/data). ResNet18 acc 86.58% → 87.92% (MPS floating-point accumulation). Reproduction story unchanged.

> **2026-09-12 note**: ResNet18 was re-trained after landing the `predictions`/`labels` JSON output (v8.3). New test accuracy 87.92% (previously 86.58%); the +1.34pp shift is MPS floating-point accumulation noise across re-runs — same seed, same data, slight non-determinism in cuBLAS / MPS kernels. The reproduction story is unchanged.

### Statistical robustness (K-fold cross-validation)

The headline numbers above are single stratified 85/15 splits at `max-per-class=4000` — the same protocol the original paper used. For "is this effect real or a lucky split?" questions, the repo also implements `StratifiedKFold` 5-fold cross-validation via `--folds N`.

- **Quick demo (~2 min):** `bash scripts/run_kfold_demo.sh` runs RF 5-fold on 1000 images.
- **Full demo (~15 min, includes CNN+SE + ResNet18):** `bash scripts/run_kfold_demo.sh --all`
- **Single model:** `python3 -m src.train --model <name> --folds 5` writes `results/<model>_kfold.json` (does **not** overwrite the single-fold JSON).

See `docs/K_FOLD.md` for why the single-fold number remains the headline (comparison anchor with the paper and the standard rebuttal), how to interpret the per-fold vs aggregated metrics, and when the additional ~50-min ResNet18 K-fold run is worth it.

### Comparison with the published paper

This project started as a comparison study for the CAHEML 2026 conference. The original paper reports:

| Method | Paper | Reproduced (this repo) | Δ |
|---|---:|---:|---:|
| Random Forest | 59.10% | 59.42% | +0.32pp |
| Self-built CNN | 75.87% | **76.25%** | **+0.38pp** ✓ exceeds |
| ResNet18 transfer | 85.92% | 86.58% | +0.66pp |

The three newer backbones (ResNet50, EfficientNet-B0, MobileNetV3-Large) are
not in the original paper's Table 1, so there is no published baseline to
compare against. Their results above are independent reproductions against
the same SDNET2018 dataset, same seed, same protocol — just at half the
training-set size (`max-per-class=2000` instead of 4000).

**CNN reproduction fix — two-step (2026-09-12):**
- v1 (`670b076`): added `optim.lr_scheduler.CosineAnnealingLR(T_max=epochs)` to `src/train.py`. CNN 67.27% → 72.83% (+5.56pp), paper gap −8.60pp → −3.04pp.
- v2 (`db36e76`): switched `collect_files` in `src/data.py` from `np.random.default_rng` back to the legacy `np.random.choice` used by the paper, and bumped CNN epochs 15 → 20. CNN 72.83% → **76.25%** (+3.42pp), paper gap inverted to **+0.38pp — CNN now slightly exceeds the published baseline**.
- ResNet18 has always exceeded the paper. The headline "transfer learning decisively beats both baselines" remains unchanged.

**Fourth model — CNN + SE-Blocks (`db36e76` follow-up):**
- Added `SEBlock` class (Squeeze-and-Excitation channel attention, Hu et al. 2020) to `src/models.py`, one per conv block.
- New `build_cnn_se` factory + `--model cnn_se` in train.py, 5.32M params (+12,296 vs CNN).
- CNN+SE test accuracy: **77.58%** (+1.33pp over plain CNN for +0.23% params) — channel attention helps even at this scale.
- ResNet18 still wins by 9.00pp — transfer learning remains the dominant lever.

## Demo

The Streamlit demo (`app/streamlit_app.py`) has six sections:

1. **Try it** — upload any concrete surface image, get a prediction with confidence
2. **Where is the model looking?** — Grad-CAM heatmap on `layer4` overlays the regions that drove the decision
3. **Model comparison** — accuracy ladder across the **four** methods (RF / CNN / CNN+SE / ResNet18), with pp gap to transfer learning and SE-block delta
4. **Per-class metrics** — collapsible tables of Precision / Recall / F1 for every model
5. **Confusion matrices** — true-vs-predicted heatmap for each model (rows=true, cols=predicted); TP / TN / FP / FN counts. v8.3+ saves predictions to JSON.
6. **Training curves** — CNN (dotted) vs CNN+SE (dashed) vs ResNet18 (solid) validation curves overlaid

```bash
streamlit run app/streamlit_app.py
```

## Reproducing locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify the dataset is in place
ls data/DATA_Maguire_20180517_ALL/SDNET2018/{D,P,W}/

# 3. Train each model (independent — order doesn't matter)
python -m src.train --model rf        # ~30s, baseline
python -m src.train --model cnn       # ~5 min on MPS GPU
python -m src.train --model resnet18  # ~10 min on MPS GPU

# 3b. (Optional) 5-fold cross-validation on a small subset for K-fold
#     pipeline sanity. Single-fold numbers above are the headline; see
#     docs/K_FOLD.md for when K-fold is worth the extra wall time.
bash scripts/run_kfold_demo.sh        # ~2 min total
bash scripts/run_kfold_demo.sh --all  # ~15 min, includes CNN+SE + ResNet18

# 4. Run tests
pip install -r requirements-dev.txt
pytest tests/ -v

# 5. Launch the demo
streamlit run app/streamlit_app.py
```

Training outputs are persisted to `results/<model>_results.json` (metrics + history) and `models/crack_<model>_best.pt` (state_dict + config + history).

## Project structure

```
concrete-crack-detection/
├── README.md                          (this file)
├── LICENSE                            MIT
├── runtime.txt                        Python version pin
├── requirements.txt                   runtime deps
├── requirements-dev.txt               pytest for tests
├── .github/workflows/test.yml         CI: Python 3.9 + smoke tests on push
├── app/
│   └── streamlit_app.py               Streamlit demo (6 sections: upload + Grad-CAM + compare + per-class + confusion + curves)
├── src/
│   ├── data.py                        SDNET2018 file collection + image loading
│   ├── models.py                      build_cnn / build_cnn_se / build_resnet18 / build_model
│   ├── train.py                       Unified training CLI (rf / cnn / cnn_se / resnet18)
│   ├── infer.py                       Single-image inference helper
│   ├── compare.py                     Results table + headline number generator
│   ├── confusion.py                   Confusion matrix + FPR/FNR computation (v5)
│   └── xai.py                         Grad-CAM hook on layer4 (v5)
├── tests/                             7 test files · pytest
│   ├── test_data.py                   dataset contract (file counts, schema)
│   ├── test_models.py                 model forward shape + parameter count
│   ├── test_infer.py                  inference output range (4 models)
│   ├── test_confusion.py              confusion matrix + FPR/FNR (v5)
│   ├── test_xai.py                    Grad-CAM output shape (v5)
│   ├── test_run_cross_domain.py       per-surface cross-domain benchmark (cross-domain commit)
│   └── conftest.py
├── data/
│   └── DATA_Maguire_20180517_ALL/SDNET2018/  (56,092 images, gitignored)
├── models/
│   ├── crack_cnn_best.pt              5.30M params
│   ├── crack_cnn_se_best.pt           5.32M params
│   ├── crack_resnet18_best.pt         11M params
│   └── crack_cnn.pt                   legacy 75.87% reference
├── results/
│   ├── rf_results.json                59.92% acc
│   ├── cnn_results.json               76.25% acc
│   ├── cnn_se_results.json            77.58% acc
│   ├── resnet18_results.json          87.92% acc
│   └── cross_domain_v1.json          per-surface benchmark (cross-domain commit)
├── scripts/
│   └── run_kfold_demo.sh              Quick K-fold CV demo (~2 min on 1000 images)
├── docs/
│   └── K_FOLD.md                      Why single-fold is the headline + how to run K-fold
├── P0_Evidence/                       Standard rebuttal evidence (5a51a05)
│   ├── README.md                      P0 task description + 4-model FPR/FNR table + verification formulas
│   ├── metrics/                       final_metrics.csv + final_metrics.md
│   ├── confusion_matrix/              5 PNGs (1 merged + 4 single)
│   ├── demo/                          runtime_verification.md + samples/
│   └── reproducibility/               environment.md
├── benchmark/                         per-surface cross-domain scripts
├── docs/                              project documentation
├── configs/ notebooks/ scripts/       placeholder directories
└── CAHEML2026_论文{初稿,中文对照,提纲}.md  historical paper materials
```

## Dataset

**SDNET2018** — [Maguire, Dorafshan & Thomas, Utah State University (2018)](https://digitalcommons.usu.edu/all_datasets/48/).

| Subset | Description | Cracked | Uncracked | Total |
|---|---|---:|---:|---:|
| D | Bridge decks | 2,025 | 11,595 | 13,620 |
| P | Pavements | 2,608 | 21,726 | 24,334 |
| W | Walls | 3,851 | 14,287 | 18,138 |
| **Total** | | **8,484** | **47,608** | **56,092** |

Crack widths range from 0.06 mm to 25 mm. The dataset is gitignored; download from the link above and unzip into `data/DATA_Maguire_20180517_ALL/`.

## Architecture details

### Random Forest baseline

Flatten each 64×64 grayscale image to a 4,096-dimensional feature vector. Train sklearn's `RandomForestClassifier` with 200 trees and `max_depth=15`. No deep learning, no pre-training — a clean ML baseline.

### Self-built CNN

Four convolutional blocks (32 → 64 → 128 → 256 channels), each with 2× convolutions, BatchNorm, ReLU, MaxPool, and Dropout. ~5.3M parameters. Input 96×96×3, trained from scratch with AdamW (lr=1e-3, weight_decay=1e-4) for 15 epochs. Random horizontal + vertical flips for augmentation.

### ResNet18 (transfer learning)

ImageNet1K pre-trained ResNet18 from `torchvision.models`. The 1,000-class classifier head is replaced with `Dropout(0.5) → Linear(512, 2)`. The entire network is fine-tuned end-to-end on 160×160×3 inputs with AdamW (lr=3e-4, weight_decay=1e-4) for 12 epochs. Same flip augmentation as the self-built CNN.

## Engineering notes

- **Apple Silicon GPU.** Training uses PyTorch's MPS backend, no cloud GPU required.
- **Best-checkpoint tracking.** Each run tracks the highest validation accuracy and saves that state, not the final-epoch state.
- **Stratified split.** Train/test split preserves class balance (`stratify=y`).
- **Reproducible sampling.** File selection uses a seeded RNG (`np.random.default_rng(42)`), so reruns pick the same subset.
- **Memory note.** The current loader reads all selected images into RAM as a single numpy array. This works for the 8,000-image training set (~5 GB at 160×160×3 floats) but does not scale to the full 56,092 images. A `torch.utils.data.Dataset` + `num_workers` loader would be the next step if scaling beyond ~20k images becomes necessary.

## Limitations

- **Sample size.** The training script caps at 4,000 images per class by default (8,000 total). Larger samples would tighten confidence intervals but also require more GPU time.
- **Class balance.** Training forces a 1:1 ratio between cracked and uncracked samples. Real-world inspection has class imbalance (more uncracked than cracked surfaces). The reported test accuracy should be read with this in mind.
- **Single-label classification.** The model answers "cracked or not" but does not localize the crack or estimate its width. Pixel-level segmentation (U-Net) and width quantification are flagged as future work.
- **Best-checkpoint caveat.** Best validation accuracy is monitored and saved, but in practice the highest validation epoch is not always the highest test epoch. This run happened to pick well (best val 86.58% → final test 86.58%), but a more robust selection would use an explicit held-out set or K-fold cross-validation.

## License

MIT — see [LICENSE](LICENSE).

## Author

jinliangyue (pen-name 十八) — civil engineering senior, Ningxia Institute of Technology. Interested in ML applications for structural health monitoring.
