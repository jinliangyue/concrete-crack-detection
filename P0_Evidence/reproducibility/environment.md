# Environment & Reproducibility

## Hardware

```
Platform:    macOS-26.6.2-arm64
Architecture: arm64 (Apple Silicon)
```

## Software Versions

| Package | Version |
|---------|---------|
| Python | 3.9.13 |
| PyTorch | 2.8.0 |
| NumPy | 2.0.2 |
| scikit-learn | 1.6.1 |
| Streamlit | 1.50.0 |
| matplotlib | 3.9.4 |
| MPS (Apple GPU) | available |

## Reproduction Commands

### Train all 4 models

```bash
cd "/Users/xiayuhao/Desktop/Claude code/项目作品/CNN裂缝检测项目"
python -m src.train --model rf        # ~30s
python -m src.train --model cnn       # ~5 min
python -m src.train --model cnn_se    # ~5 min
python -m src.train --model resnet18  # ~10 min
```

### Run tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

### Launch Streamlit demo

```bash
streamlit run app/streamlit_app.py
```

## Data

| Subset | Description | Path |
|--------|-------------|------|
| SDNET2018 | 56,092 images · bridge decks / pavements / walls | `data/DATA_Maguire_20180517_ALL/SDNET2018/` (gitignored) |

## Model Checkpoints (gitignored)

| Model | File | Size |
|-------|------|------|
| Random Forest | n/a (sklearn) | n/a |
| Self-built CNN | `models/crack_cnn_best.pt` | ~20 MB |
| CNN + SE-Blocks | `models/crack_cnn_se_best.pt` | ~20 MB |
| ResNet18 | `models/crack_resnet18_best.pt` | ~45 MB |

## Results Files

| File | Description |
|------|-------------|
| `results/rf_results.json` | RF metrics + predictions + labels |
| `results/cnn_results.json` | CNN metrics + history + predictions + labels |
| `results/cnn_se_results.json` | CNN+SE metrics + history + predictions + labels |
| `results/resnet18_results.json` | ResNet18 metrics + history + predictions + labels |

## Reproducibility

- Random seed: 42
- PyTorch MPS deterministic: enabled per run config
- File selection RNG: `np.random.default_rng(42)`

## Git Status

```
local repo: yes (.git/ present in CNN裂缝检测项目/)
github: jinliangyue/concrete-crack-detection (last README mentions)
```

Streamlit deployment: `concrete-crack-detection.streamlit.app` (per README badge)