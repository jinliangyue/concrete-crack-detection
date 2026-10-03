"""
Cross-domain evaluation · SDNET2018 surface split
==================================================

Loads the three trained checkpoints (CNN, CNN+SE, ResNet18) and runs
zero-shot inference on each SDNET2018 *surface* subset independently
(D = bridge deck, P = pavement, W = wall).

The training set covers all three surfaces mixed; this script measures
how well each model generalises when evaluated on a single surface at
a time. It is not a fresh train/test split — it is an evaluation-only
script that uses the existing checkpoints as feature classifiers.

Why this counts as "external" relative to the project's normal eval
- The original test split draws samples proportionally from D / P / W
  (mixed). This script separates them so the per-surface accuracy is
  visible.
- The data is the *same* SDNET2018, but the per-surface subset the
  model has never been evaluated on in isolation is a real distribution
  shift — pavement texture vs. wall texture vs. deck texture is
  genuinely different, even with the same labels.

For a strictly external dataset (e.g. Mendeley Concrete Crack Images
by Çağlar Özgenel), a parallel script lives next to this one in
`benchmark/external/mendeley/`; it is opt-in because it requires an
~600 MB download.

Usage
-----
    cd <project root>
    python -m benchmark.external.run_cross_domain                 # all 3 models × 3 surfaces
    python -m benchmark.external.run_cross_domain --models resnet18
    python -m benchmark.external.run_cross_domain --surfaces D
    python -m benchmark.external.run_cross_domain --max-per-class 200 --seed 42
    python -m benchmark.external.run_cross_domain --save-json results/cross_domain.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Make `src.*` importable when running from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    collect_files,
    load_images,
    normalize_imagenet,
)
from src.models import build_cnn, build_cnn_se, build_resnet18  # noqa: E402


# --------------------------------------------------------------------- model zoo
CHECKPOINTS: Dict[str, Dict] = {
    "cnn": {
        "path": PROJECT_ROOT / "models" / "crack_cnn_best.pt",
        "img_size": 96,
        "loader": lambda: build_cnn(img_size=96),
    },
    "cnn_se": {
        "path": PROJECT_ROOT / "models" / "crack_cnn_se_best.pt",
        "img_size": 96,
        "loader": lambda: build_cnn_se(img_size=96),
    },
    "resnet18": {
        "path": PROJECT_ROOT / "models" / "crack_resnet18_best.pt",
        "img_size": 160,
        "loader": lambda: build_resnet18(pretrained=False),
    },
}

SURFACE_LABELS = {
    "D": "Bridge Deck",
    "P": "Pavement",
    "W": "Wall",
}


# --------------------------------------------------------------------- IO
def _load_model(name: str) -> Tuple[torch.nn.Module, int, torch.device]:
    """Build model from `name`, load weights, return (model, img_size, device).

    Uses MPS on Apple Silicon when available, otherwise CPU. The
    inference path inside `_predict` honours the returned device.
    """
    cfg = CHECKPOINTS[name]
    if not cfg["path"].exists():
        raise FileNotFoundError(
            f"checkpoint for {name!r} not found at {cfg['path']}. "
            f"Train it first or pass --models with only available checkpoints."
        )
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    ckpt = torch.load(cfg["path"], map_location=device, weights_only=False)
    model = cfg["loader"]()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    img_size = ckpt.get("config", {}).get("img_size", cfg["img_size"])
    return model, img_size, device


def _load_surface(surface: str, max_per_class: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load images + labels for one surface (D / P / W).

    Returns (X, y) where X has shape (N, 3, H, W) float32 in [0, 1]
    and y has shape (N,) with 0 = NoCrack, 1 = Crack.
    """
    files_crack, files_nocrack = collect_files(
        max_per_class=max_per_class, surfaces=[surface], seed=seed,
    )
    print(
        f"  [{surface}] {len(files_crack)} cracked + {len(files_nocrack)} uncracked "
        f"= {len(files_crack) + len(files_nocrack)} total"
    )
    img_size = max(cfg["img_size"] for cfg in CHECKPOINTS.values())
    # All three models share normalisation once we normalise to ImageNet;
    # but the spatial size differs (96 vs 160). Load at the *largest*
    # required size (160); the CNN/SE branch in `_predict` downsamples
    # back to 96 inside the loop.
    return load_images(
        files_crack + files_nocrack,
        img_size=img_size,
        grayscale=False,
    )


# --------------------------------------------------------------------- inference
@torch.no_grad()
def _predict(
    model: torch.nn.Module, X: np.ndarray, img_size: int,
    device: torch.device, batch_size: int = 64,
) -> np.ndarray:
    """Run forward pass, return predicted labels (N,)."""
    # Per-model resize: the CNN expects 96×96, ResNet18 expects 160×160.
    # We loaded at 160, so downsample to 96 for CNN/SE.
    if X.shape[-1] != img_size:
        from PIL import Image
        resized = np.empty((X.shape[0], 3, img_size, img_size), dtype=np.float32)
        for i in range(X.shape[0]):
            arr = (X[i].transpose(1, 2, 0) * 255.0).astype("uint8")
            img = Image.fromarray(arr).resize((img_size, img_size))
            resized[i] = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        X = resized

    X_n = normalize_imagenet(X)
    preds: List[int] = []
    for start in range(0, len(X_n), batch_size):
        batch = torch.from_numpy(X_n[start : start + batch_size]).float().to(device)
        out = model(batch)
        preds.extend(out.argmax(dim=1).cpu().tolist())
    return np.asarray(preds, dtype=np.int64)


# --------------------------------------------------------------------- main loop
def evaluate_one(
    model_name: str,
    surface: str,
    X: np.ndarray,
    y: np.ndarray,
    max_per_class: int,
    seed: int,
) -> Dict:
    """Run one (model, surface) pair end-to-end."""
    model, img_size, device = _load_model(model_name)
    started = time.perf_counter()
    preds = _predict(model, X, img_size, device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return {
        "model": model_name,
        "surface": surface,
        "surface_label": SURFACE_LABELS[surface],
        "n_total": int(len(y)),
        "n_crack": int(y.sum()),
        "n_uncracked": int(len(y) - y.sum()),
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
        "elapsed_ms": round(elapsed_ms, 1),
        "max_per_class": max_per_class,
        "seed": seed,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--models", nargs="+", default=list(CHECKPOINTS),
        choices=list(CHECKPOINTS),
        help="Models to evaluate (default: all available)",
    )
    parser.add_argument(
        "--surfaces", nargs="+", default=list(SURFACE_LABELS),
        choices=list(SURFACE_LABELS),
        help="SDNET2018 surfaces to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=500,
        help="Sample N images per class per surface (default 500; total = 6N × 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for the per-class sample",
    )
    parser.add_argument(
        "--save-json", type=Path, default=None,
        help="Optional path to save the per-(model, surface) results JSON",
    )
    args = parser.parse_args(argv)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Pre-load each surface once; cache by surface name.
    surface_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for surface in args.surfaces:
        print(f"Loading surface {surface} ({SURFACE_LABELS[surface]})...")
        surface_data[surface] = _load_surface(surface, args.max_per_class, args.seed)

    results: List[Dict] = []
    print()
    print(f"{'model':<10} {'surface':<8} {'n':>5} {'acc':>6} {'P':>6} {'R':>6} {'F1':>6} {'ms':>7}")
    print("-" * 60)
    for model_name in args.models:
        try:
            _load_model(model_name)  # fail fast with a clear message
        except FileNotFoundError as exc:
            print(f"  [skip] {exc}")
            continue
        for surface in args.surfaces:
            X, y = surface_data[surface]
            r = evaluate_one(
                model_name, surface, X, y,
                max_per_class=args.max_per_class, seed=args.seed,
            )
            results.append(r)
            print(
                f"{r['model']:<10} {r['surface']:<8} {r['n_total']:>5} "
                f"{r['accuracy']:>6.4f} {r['precision']:>6.4f} "
                f"{r['recall']:>6.4f} {r['f1']:>6.4f} {r['elapsed_ms']:>7.0f}",
                flush=True,
            )

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w", encoding="utf-8") as fp:
            json.dump(
                {
                    "config": {
                        k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()
                    },
                    "results": results,
                },
                fp,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nSaved {len(results)} rows to {args.save_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
