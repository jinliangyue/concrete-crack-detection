"""
Compare accuracy across the three trained models.

Reads results/<model>_results.json files and prints a comparison table.
Used by the Streamlit demo and the README "三组对比" section.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.data import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_LABELS = {
    "rf": "Random Forest (pixel features)",
    "cnn": "Self-built CNN (4 layers, ~5.3M params)",
    "resnet18": "ResNet18 (ImageNet transfer learning)",
}

# Headline numbers from the original experiments (used when JSON is not yet present)
LEGACY_NUMBERS = {
    "rf":       {"accuracy": 0.5910, "n_test": 1000, "n_train": 4000},
    "cnn":      {"accuracy": 0.7587, "n_test": 1500, "n_train": 8500},
    "resnet18": {"accuracy": 0.8592, "n_test": 1200, "n_train": 6800},
}


def load_results() -> Dict[str, dict]:
    """Load JSON results for each model. Falls back to legacy headline numbers."""
    out: Dict[str, dict] = {}
    for name in ("rf", "cnn", "resnet18"):
        path = RESULTS_DIR / f"{name}_results.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            out[name] = {
                "accuracy": data.get("accuracy", LEGACY_NUMBERS[name]["accuracy"]),
                "n_train": data.get("config", {}).get("n_train", LEGACY_NUMBERS[name]["n_train"]),
                "n_test": data.get("config", {}).get("n_test", LEGACY_NUMBERS[name]["n_test"]),
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
    for key in ("rf", "cnn", "resnet18"):
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
    """Returns 'X → Y → Z' format string showing the accuracy ladder."""
    rows = comparison_table()
    return " → ".join(f"{r['accuracy_pct']:.2f}%" for r in rows)


if __name__ == "__main__":
    print("SDNET2018 裂缝检测三组方法对比\n")
    print(f"{'Method':<45} {'Accuracy':>10} {'Train':>8} {'Test':>8} {'Source':<25}")
    print("-" * 100)
    for row in comparison_table():
        print(
            f"{row['model']:<45} {row['accuracy_pct']:>9.2f}% "
            f"{row['n_train']:>8} {row['n_test']:>8} {row['source']:<25}"
        )
    print(f"\nAccuracy ladder: {headline_improvement()}")
