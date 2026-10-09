#!/usr/bin/env python3
"""
Generate the README "results comparison" plot + per-model confusion
matrices from results/*.json.

Outputs (under docs/figures/):
    results_comparison.png     — 7-model accuracy + FPR bar chart (headline K-fold)
    confusion_<model>.png      — per-model normalized confusion matrix
                                (for ResNet18 + CNN+SE + MobileNetV3-Large,
                                 using saved predictions + labels)

Run:
    python3 scripts/plot_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# Headline K-fold numbers (all max-per-class=4000).
MODELS_KFOLD = [
    ("RF",                59.12, 1.00, 46.0, 2.4, 35.7),
    ("CNN",               None,  None, None, None, None),  # not run on K-fold (headline only)
    ("CNN + SE-Blocks",   72.04, 2.21, 26.7, 4.2, 29.3),
    ("ResNet18",          87.44, 0.74,  9.1, 1.8, 16.1),
    ("ResNet50",          None,  None, None, None, None),  # trained on 2000/class
    ("EfficientNet-B0",   None,  None, None, None, None),  # trained on 2000/class
    ("MobileNetV3-Large", None,  None, None, None, None),  # trained on 2000/class
]

# Single-fold headline numbers (always available, used for fallback bar).
MODELS_SINGLE_FOLD = {
    "RF":                59.92,
    "CNN":               76.25,
    "CNN + SE-Blocks":   77.58,
    "ResNet18":          87.92,
    "ResNet50":          84.00,
    "EfficientNet-B0":   85.17,
    "MobileNetV3-Large": 86.67,
}


def plot_results_comparison() -> Path:
    """7-model bar chart: single-fold accuracy (all) + K-fold mean ± std
    (where available). One chart, two bar series per model."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    labels = [name for name, *_ in MODELS_KFOLD]
    sf = [MODELS_SINGLE_FOLD[name] for name in labels]
    kf_mean = [m[1] for m in MODELS_KFOLD]
    kf_std = [m[2] for m in MODELS_KFOLD]

    x = np.arange(len(labels))
    width = 0.36

    bars_sf = ax.bar(x - width / 2, sf, width,
                    label="Single fold (max-per-class=4000)", color="#4C72B0")
    bars_kf_mean = [v if v is not None else 0 for v in kf_mean]
    bars_kf = ax.bar(x + width / 2, bars_kf_mean, width,
                    yerr=[(s if s is not None else 0) for s in kf_std],
                    label="5-fold CV mean ± std (headline data only)",
                    color="#DD8452", alpha=0.92, capsize=3)

    # Annotate each bar with its value.
    for bar, value in zip(bars_sf, sf):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{value:.2f}%", ha="center", va="bottom", fontsize=9, color="#4C72B0")
    for bar, value, std in zip(bars_kf, kf_mean, kf_std):
        if value is None:
            ax.text(bar.get_x() + bar.get_width() / 2, 1.2,
                    "n/a", ha="center", va="bottom", fontsize=9, color="#A0522D", alpha=0.5)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2,
                    f"{value:.2f}%", ha="center", va="bottom", fontsize=9, color="#A0522D")

    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(50, 100)
    ax.set_title("Concrete crack detection — model comparison (SDNET2018)",
                 fontsize=12, pad=14)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = FIGURES_DIR / "results_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}")
    return out


def plot_confusion_matrix(model: str, json_path: Path) -> Path | None:
    """2x2 normalized confusion matrix from saved predictions + labels."""
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text())
    preds = data.get("predictions")
    labels = data.get("labels")
    if preds is None or labels is None:
        return None

    y_true = np.asarray(labels)
    y_pred = np.asarray(preds)
    # Rows = true class, Cols = predicted class; order [NoCrack=0, Crack=1].
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["NoCrack", "Crack"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["NoCrack", "Crack"])
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(f"{model} (n={len(y_true)})", fontsize=11)

    # Annotate cells.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                    ha="center", va="center",
                    color=("white" if cm_norm[i, j] > 0.5 else "black"),
                    fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = FIGURES_DIR / f"confusion_{model}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}")
    return out


def main() -> int:
    print("Generating README figures...")
    plot_results_comparison()
    for model, json_name in [
        ("resnet18",          "resnet18_results.json"),
        ("cnn_se",            "cnn_se_results.json"),
        ("mobilenetv3_large", "mobilenetv3_large_results.json"),
    ]:
        plot_confusion_matrix(model, RESULTS_DIR / json_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())