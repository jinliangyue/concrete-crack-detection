# External / Cross-domain Benchmark · Concrete Crack Detection

This directory hosts **third-party-data** and **cross-domain** evaluations
for the four trained models (RF, CNN, CNN+SE, ResNet18). The goal is to
defend the headline accuracy numbers against the "you only tested on one
split of one dataset" critique.

## Why this matters

The headline numbers (RF 59.92% / CNN 76.25% / CNN+SE 77.58% / ResNet18
87.92%) come from a single stratified split of **SDNET2018** (Maguire
et al., 2018). That split mixes the three SDNET2018 surfaces (bridge
deck, pavement, wall) proportionally. A reasonable reviewer question is:

> "How do these numbers behave when the surface distribution shifts?"

`run_cross_domain.py` answers that question without training new
models: it loads each saved checkpoint and evaluates it on **one surface
at a time**, so the per-surface accuracy is visible.

## What's here

| File | What it does |
|---|---|
| `run_cross_domain.py` | Loads the 3 best checkpoints, evaluates each on each SDNET2018 surface independently. |
| `mendeley/` (placeholder) | Reserved for the optional, opt-in **Mendeley Concrete Crack Images** evaluation. |
| `results/cross_domain_*.json` | Per-run output snapshots (committed after each release). |

## Reproducing the baseline run

```bash
# All models × all surfaces, default 500 images per class per surface
python -m benchmark.external.run_cross_domain --save-json results/cross_domain_v1.json
```

This takes ~2-3 minutes on Apple Silicon (MPS) for ResNet18 and under
a minute for the two CNN variants. Latency breakdown is part of the
JSON output.

### Smaller smoke run (≈ 10 seconds)

```bash
python -m benchmark.external.run_cross_domain \
    --max-per-class 50 --models resnet18 --surfaces D
```

## SDNET2018 surfaces

```
SDNET2018/
├── D/        Bridge Deck
│   ├── CD/   Cracked
│   └── UD/   Uncracked
├── P/        Pavement
│   ├── CP/   Cracked
│   └── UP/   Uncracked
└── W/        Wall
    ├── CW/   Cracked
    └── UW/   Uncracked
```

The training set draws from all six subfolders. Evaluating on a single
surface at a time probes how robust each model's learned features are
to texture / lighting / scale shifts.

## Mendeley Concrete Crack Images (opt-in)

A second public dataset, published by Çağlar Özgenel
([Mendeley Data, DOI: 10.17632/5y9wdsg2zt.2](https://data.mendeley.com/datasets/5y9wdsg2zt)),
contains 458 high-resolution images (200 cracked + 258 uncracked) of
concrete walls. It is genuinely external to SDNET2018 — different
sensor, different building type, different country.

To run the Mendeley evaluation:

```bash
# 1. Download (~600 MB)
python -m benchmark.external.mendeley.download

# 2. Run cross-dataset eval
python -m benchmark.external.mendeley.run_eval
```

Both scripts and a documented `README.md` for that workflow are stubbed
in this directory but **not wired up yet** — the connection depends on
whether the runtime can reach `data.mendeley.com`. The README at the
top of `mendeley/` will spell out the exact steps once that script is
shipped.

## What "external" means here

| Evaluation | External? | Reason |
|---|---|---|
| SDNET2018 mixed split (original) | No | Same dataset, single mixed-stratify split. |
| SDNET2018 per-surface (this script) | Sort of | Same dataset, but each surface has its own texture / lighting distribution; the model was trained on all surfaces mixed. |
| Mendeley Concrete Crack Images | Yes | Different images, different building type, different publication. |

This mirrors the BuildHealth NAB (Numenta Anomaly Benchmark) cross-data
story: same architecture question, applied to a different public
dataset that is genuinely outside the project's training distribution.
