"""
Compare accuracy across the four trained models.

Reads results/<model>_results.json files and prints a comparison table.
Used by the Streamlit demo and the README "四组对比" section.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.data import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_LABELS = {
    "rf":       "Random Forest (pixel features)",
    "cnn":      "Self-built CNN (4 layers, ~5.3M params)",
    "cnn_se":   "Self-built CNN + SE-Blocks (channel attention, ~5.31M params)",
    "resnet18": "ResNet18 (ImageNet transfer learning)",
}

# Headline numbers from the original experiments (used when JSON is not yet present)
LEGACY_NUMBERS = {
    "rf":       {"accuracy": 0.5910, "n_test": 1000, "n_train": 4000},
    "cnn":      {"accuracy": 0.7587, "n_test": 1500, "n_train": 8500},
    "cnn_se":   {"accuracy": 0.0,    "n_test": 1200, "n_train": 6800},  # No paper baseline — must come from JSON
    "resnet18": {"accuracy": 0.8592, "n_test": 1200, "n_train": 6800},
}


def load_results() -> Dict[str, dict]:
    """Load JSON results for each model. Falls back to legacy headline numbers."""
    out: Dict[str, dict] = {}
    for name in ("rf", "cnn", "cnn_se", "resnet18"):
        path = RESULTS_DIR / f"{name}_results.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            legacy = LEGACY_NUMBERS[name]
            out[name] = {
                "accuracy": data.get("accuracy", legacy["accuracy"]),
                "n_train": data.get("config", {}).get("n_train", legacy["n_train"]),
                "n_test": data.get("config", {}).get("n_test", legacy["n_test"]),
                "source": "results/" + path.name,
            }
        else:
            out[name] = {
                **LEGACY_NUMBERS[name],
                "source": "headline (no JSON yet)",
            }
    return out


def comparison_table() -> List[Dict]:
    """Return rows for tabular display."""
    results = load_results()
    rows = []
    for key in ("rf", "cnn", "cnn_se", "resnet18"):
        r = results[key]
        rows.append({
            "model": MODEL_LABELS[key],
            "key": key,
            "accuracy_pct": round(r["accuracy"] * 100, 2),
            "n_train": r["n_train"],
            "n_test": r["n_test"],
            "source": r["source"],
        })
    return rows


def headline_improvement() -> str:
    """Returns 'X → Y → Z → W' format string showing the accuracy ladder."""
    rows = comparison_table()
    return " → ".join(f"{r['accuracy_pct']:.2f}%" for r in rows)


if __name__ == "__main__":
    print("SDNET2018 裂缝检测四组方法对比\n")
    print(f"{'Method':<55} {'Accuracy':>10} {'Train':>8} {'Test':>8} {'Source':<25}")
    print("-" * 110)
    for row in comparison_table():
        print(
            f"{row['model']:<55} {row['accuracy_pct']:>9.2f}% "
            f"{row['n_train']:>8} {row['n_test']:>8} {row['source']:<25}"
        )
    print(f"\nAccuracy ladder: {headline_improvement()}")
