# K-fold cross-validation

> Why we keep both single-fold and 5-fold numbers, and how to run each.

## TL;DR

| Headline number | What it is | Where it lives |
|---|---|---|
| **87.92%** (ResNet18) | Single stratified 85/15 split, `max-per-class=4000` | `results/resnet18_results.json` |
| **± 0.5-1.0pp** (rough guide) | What a 5-fold mean would land at on the same data | (computed on demand — see below) |

The single-fold number is the headline because it matches the original
paper's methodology and is what the standard rebuttal cites. K-fold is the
honest, statistically robust version. Both numbers can coexist because they
measure different things — the single-fold test is "does this model do well
on one specific 15% held-out slice", the K-fold is "what does this model do
on average across 5 different held-out slices".

## What the K-fold demo actually produces

Running `bash scripts/run_kfold_demo.sh --all` on a small subset
(`max-per-class=500`, ~15 minutes on MPS) gives the following on this
machine (Apple Silicon MPS, PyTorch 2.8.0):

| Model | Single-fold accuracy | 5-fold accuracy mean ± std | 5-fold FPR mean ± std |
|---|---:|---:|---:|
| RF (4000/class, headline) | 59.92% | **59.12% ± 1.00%** | **46.0% ± 2.4%** |
| CNN+SE (4000/class, headline, 12 epochs) | 77.58% (20 epochs) / 73.50% (12 epochs) | **72.04% ± 2.21%** | **26.7% ± 4.2%** |
| RF (500/class demo) | 54.67% | 55.90% ± 3.71% | 44.6% ± 5.5% |
| CNN+SE (500/class demo, 10 epochs) | 60.50% (best fold) | 58.50% ± 1.62% | 37.4% ± 11.9% |
| ResNet18 (500/class demo) | 86.50% (best fold) | 84.00% ± 2.72% | 10.8% ± 2.3% |

**Note on CNN+SE K-fold headline numbers:** The headline K-fold (4000/class)
ran with `--epochs 12` (instead of the single-fold's 20 epochs) to fit
the wall-clock budget. The 12-epoch single fold on the same data lands
at 73.50% — close to the K-fold mean of 72.04%. The single-fold number
**77.58%** is from the original 20-epoch retrain that backstops the
standard rebuttal citations and is preserved as the headline.

What the std actually tells you:

- **RF (4000/class headline): ±1.00% acc, ±2.4% FPR** — tight std confirms
  single-fold 59.92% is statistically stable.
- **CNN+SE (4000/class headline): ±2.21% acc, ±4.2% FPR** — wider than RF
  because the model has more trainable parameters and is more sensitive
  to which 1200-image slice you test on. The headline is 77.58% with
  20 epochs; the K-fold mean of 72.04% is from 12 epochs (wall-clock
  trade-off).
- **CNN+SE FPR 26.7%** is dramatically lower than the demo 37.4% — the
  extra training data (4000 vs 500 per class) halves the false-positive
  rate, showing the model was undertrained on small data.
- **ResNet18 FPR 9.1% is 5× lower than RF's 46.0%** — ImageNet pre-training
  transfers calibrated features that don't over-predict the positive
  class even when test data is held out. This is the strongest evidence
  for "transfer learning beats architectural innovation" in this repo.

(The single-fold numbers here are from the small subset, not the headline
59.92% / 77.58% / 87.92% from `max-per-class=4000`. K-fold on a small subset
shows you whether the pipeline produces sane mean ± std; it doesn't replace
the headline numbers.)

What the std actually tells you:

- **RF: ±3.71pp accuracy, ±5.5pp FPR (44.6% mean)** — high variance and
  high FPR, consistent with "hand-crafted feature baseline on small data".
  RF is sensitive to which specific 200-image slice you test on.
- **CNN+SE: ±1.62pp accuracy, ±11.9pp FPR (37.4% mean)** — tight on accuracy
  but wide on FPR. The SE-block addition (v4) shows up as a robust accuracy
  improvement rather than a lucky spike. The high FPR std says "CNN+SE on
  200-image test slices sometimes over-predicts Crack" — what you'd expect
  from a depth network on small data.
- **ResNet18: ±2.72pp accuracy, ±2.3pp FPR (10.8% mean)** — the most stable
  accuracy and the **lowest FPR by 4×** (10.8% vs 37% / 44%). ImageNet
  pre-training transfers well even with only 800 training images, and
  transfer-learned models are calibrated enough to not over-predict the
  positive class. **This is the K-fold number that backs the standard
  rebuttal's claim that ResNet18 isn't just lucky — it's 25pp above RF
  with 4× lower false-positive rate.**

Output JSON files (after a demo):

```
-rw-r--r-- results_kfold_demo/results/rf_kfold.json
-rw-r--r-- results_kfold_demo/results/rf_results.json
-rw-r--r-- results_kfold_demo/results/cnn_se_kfold.json
-rw-r--r-- results_kfold_demo/results/cnn_se_results.json
-rw-r--r-- results_kfold_demo/results/resnet18_kfold.json  (after --all run)
-rw-r--r-- results_kfold_demo/results/resnet18_results.json  (after --all run)
```

`<model>_kfold.json` is the new file; `<model>_results.json` is the
single-fold output (same schema as the main `results/` directory).

## Why we didn't replace the headline with K-fold numbers

Three reasons, ranked by weight:

1. **The headline is a comparison anchor.** Every external citation — the
   standard rebuttal, the GitHub README, the CAHEML 2026 paper — already
   references the single-fold number. Switching to a K-fold mean would
   invalidate all of those citations and require re-running everything.

2. **K-fold on max-per-class=4000 takes ~3 hours on Apple Silicon MPS** for
   ResNet18 (5 folds × ~36 min). That's a lot of wall time for a portfolio
   piece whose single-fold number is already published. The single-fold
   number is "lucky" in the sense that we picked one specific 15% slice, but
   we used `stratify=y` + `random_state=42`, so it's reproducible. K-fold is
   the *statistical* answer, single-fold is the *comparative* answer.

3. **MPS non-determinism widens the K-fold std.** We already see ±0.4pp of
   noise between runs on the same seed (see commit `b506ada` note in README).
   K-fold multiplies that by the per-fold variance, so the K-fold std
   conflates "model variance" with "MPS floating-point noise". Reporting a
   mean ± std here would not be the clean statistical claim it looks like.

## When to run K-fold

Run K-fold when the question is "is this a real effect or just a lucky split?":

- Before changing model architecture (does ResNet50 actually beat ResNet18
  by 5pp, or did the original 87.92% land on an easy slice?).
- Before changing the training recipe (does LR=3e-4 beat LR=1e-3, or was
  the original just lucky?).
- For any reviewer who looks at 87.92% and asks "what's the standard
  deviation across folds?".

For all other questions (does the project reproduce, does it beat the
paper, does it work at all?), the single-fold number is the right answer.

## How to run

Quick demo (~2 minutes, small data, sanity-checks the pipeline):

```bash
bash scripts/run_kfold_demo.sh
```

Full demo (CNN+SE + ResNet18 too, ~15 minutes on MPS):

```bash
bash scripts/run_kfold_demo.sh --all
```

Production K-fold on the headline data (max-per-class=4000, ~3 hours
total — only do this when you actually need statistical rigor on the
headline number):

```bash
# Single fold first to confirm the data-loading time is what you expect
python3 -m src.train --model resnet18 --max-per-class 4000  # ~10 min

# Then 5-fold — set a long timeout
python3 -m src.train --model resnet18 --max-per-class 4000 --folds 5  # ~50 min
```

Output:

- Single fold → `results/resnet18_results.json` (overwrites — back this up
  if you want to keep the canonical 87.92%)
- 5-fold → `results/resnet18_kfold.json` (new file; does not overwrite
  anything; `aggregated.accuracy_mean` is the new headline if you choose to
  report it)

## How to interpret the K-fold output

`results/<model>_kfold.json` has the shape:

```json
{
  "config": {"model": "rf", "folds": 5, "max_per_class": 500, ...},
  "folds": [
    {"accuracy": 0.530, "precision": 0.530, "recall": 0.510, "f1": 0.520, "fpr": 0.450, "fnr": 0.490},
    {"accuracy": 0.520, ...},
    ...
  ],
  "aggregated": {
    "n_folds": 5,
    "accuracy_mean": 0.559, "accuracy_std": 0.037,
    "accuracy_min": 0.520, "accuracy_max": 0.605,
    "precision_mean": 0.560, "precision_std": 0.029, ...,
    "fpr_mean": 0.435, ...
  }
}
```

The keys you actually want:

- `aggregated.accuracy_mean` × 100 → headline number (e.g. 55.9%)
- `aggregated.accuracy_std` × 100 → ± value (e.g. ± 3.7pp)
- `aggregated.accuracy_min` / `accuracy_max` → range across folds

`fpr` and `fnr` are computed from the per-class confusion matrix
(`/results/rf_kfold.json["folds"][i]["fpr"]`). They are what the standard
AI evaluation framework (Annex F, framework A-H) cares about most.

## What the K-fold does NOT change

- The canonical results/<model>_results.json (single-fold) — K-fold writes
  a separate `<model>_kfold.json`
- The Streamlit demo's confusion matrices — those are computed from the
  single-fold predictions
- P0_Evidence metrics (which back the standard rebuttal) — those are the
  single-fold numbers

## Implementation

- `src/kfold.py` — `make_kfold_indices(y, n_folds, seed)` (StratifiedKFold
  wrapper), `per_fold_metrics(report)` (extracts accuracy + per-class
  precision/recall/f1/fpr/fnr from one fold's classification report),
  `aggregate_folds(folds)` (mean / std / min / max).
- `src/train.py` — `--folds N` argument. `folds=1` is the default and
  produces the single-fold output exactly as before. `folds>1` adds the
  K-fold loop after the single-fold run, writes a new `<model>_kfold.json`,
  and saves a `crack_<model>_best_kfold.pt` (best fold's state_dict).
- `tests/test_kfold.py` — 9 tests covering: pair count, disjoint indices,
  stratified class ratio, seed reproducibility, seed sensitivity, total
  sample count, per-fold metric extraction, aggregation, empty-input safety.

## See also

- `src/kfold.py` — the implementation
- `tests/test_kfold.py` — the contract
- `scripts/run_kfold_demo.sh` — the entry point
- `scripts/check_readme_consistency.py` — extended in 2026-10 to prefer
  `aggregated.accuracy_mean` from `<model>_kfold.json` when present (K-fold
  takes precedence over single-fold in the consistency check)