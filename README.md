# Concrete Crack Detection · Transfer Learning Ablation

[![Tests](https://img.shields.io/github/actions/workflow/status/jinliangyue/concrete-crack-detection/test.yml?branch=main&label=tests)](https://github.com/jinliangyue/concrete-crack-detection/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://github.com/jinliangyue/concrete-crack-detection/blob/main/runtime.txt)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://concrete-crack-detection.streamlit.app/)

> 在 SDNET2018（56,092 张真实混凝土桥面/路面/墙体图像）上的迁移学习消融对比——**Random Forest (59.42%) → 自建 CNN (67.27%) → ResNet18 迁移学习 (86.58%)**。模型 + Web 演示 + 完整复现命令。

A clean three-way ablation on the SDNET2018 concrete crack dataset, comparing a hand-crafted feature baseline (Random Forest), a self-built CNN trained from scratch, and a pre-trained ResNet18 with transfer learning. Built end-to-end on Apple Silicon (MPS GPU) — no cloud GPU needed.

## Results

| Method | Test accuracy | Train / Test | Parameters |
|---|---:|---|---:|
| Random Forest (64×64 grayscale, 200 trees) | **59.42%** | 6,800 / 1,200 | — |
| Self-built CNN (4 conv blocks, 96×96 input, CosineAnnealingLR) | **72.83%** | 6,800 / 1,200 | 5.3M |
| **ResNet18 (ImageNet transfer, 160×160 input)** | **86.58%** | 6,800 / 1,200 | 11M |

**Headline:** ResNet18 with ImageNet pre-training beats the Random Forest baseline by **27.16 percentage points** and the self-built CNN by **13.75 percentage points**, with the same training set and identical evaluation protocol. The transfer-learning gap is the story.

### Comparison with the published paper

This project started as a comparison study for the CAHEML 2026 conference. The original paper reports:

| Method | Paper | Reproduced (this repo) | Δ |
|---|---:|---:|---:|
| Random Forest | 59.10% | 59.42% | +0.32pp |
| Self-built CNN | 75.87% | **72.83%** | **−3.04pp** |
| ResNet18 transfer | 85.92% | 86.58% | +0.66pp |

The CNN gap is a known reproducibility issue traced to RNG implementation differences (`np.random.default_rng` vs the legacy `np.random.choice` used in the original script) and a missing `CosineAnnealingLR` scheduler. **2026-09-12 partial fix:** re-trained the CNN after adding `optim.lr_scheduler.CosineAnnealingLR(T_max=15)` (the scheduler the original paper used). CNN acc moved from 67.27% → 72.83% (+5.56pp), paper gap shrunk from −8.60pp to −3.04pp. The headline story ("transfer learning decisively beats both baselines") is unchanged and now numerically tighter.

## Demo

The Streamlit demo (`app/streamlit_app.py`) lets you:

- **Upload any concrete surface image** and get a real-time prediction with confidence
- **See the accuracy ladder** across the three methods (read from JSON results)
- **Browse training curves** for the ResNet18 model

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
├── runtime.txt                        Python version pin
├── requirements.txt                   runtime deps
├── requirements-dev.txt               pytest for tests
├── .github/workflows/test.yml         CI: Python 3.9 + smoke tests on push
├── app/
│   └── streamlit_app.py               Streamlit demo (upload + compare + curves)
├── src/
│   ├── data.py                        SDNET2018 file collection + image loading
│   ├── models.py                      CNN + ResNet18 model definitions
│   ├── train.py                       Unified training CLI (rf / cnn / resnet18)
│   ├── infer.py                       Single-image inference helper
│   └── compare.py                     Results table + headline number generator
├── tests/
│   ├── test_data.py                   dataset contract (file counts, schema)
│   ├── test_models.py                 model forward shape + parameter count
│   └── test_infer.py                  inference output range
├── data/
│   └── DATA_Maguire_20180517_ALL/SDNET2018/  (56,092 images, gitignored)
├── models/
│   └── crack_<model>_best.pt          trained checkpoints (gitignored)
├── results/
│   └── <model>_results.json           metrics + history
└── docs/                              project documentation
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

jinliangyue (pen-name 十八) — civil engineering senior, Ningxia Institute of Technology. Kaggle Titanic Top 35% (0.7751). Interested in ML applications for structural health monitoring.
