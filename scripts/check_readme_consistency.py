#!/usr/bin/env python3
"""
Cross-check README headline numbers against the canonical results JSONs.

Catches the class of bug we saw on 2026-10-08: README L8 headline still
listed pre-re-train numbers (59.42% / 67.27% / 86.58%) while the Results
table held the post-re-train numbers (59.92% / 76.25% / 87.92%). The
single-source-of-truth for headline numbers is results/<model>_results.json.

Usage:
    python scripts/check_readme_consistency.py
    # Exit code 0 = consistent, 1 = drift detected
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "README.md"
RESULTS_DIR = PROJECT_ROOT / "results"

# (model, headline label in README, accuracy key in results JSON)
MODELS = [
    ("rf",                "Random Forest",   "rf_results.json"),
    ("cnn",               "Self-built CNN",  "cnn_results.json"),
    ("cnn_se",            "CNN + SE-Blocks", "cnn_se_results.json"),
    ("resnet18",          "ResNet18",        "resnet18_results.json"),
    ("resnet50",          "ResNet50",        "resnet50_results.json"),
    ("efficientnet_b0",   "EfficientNet-B0", "efficientnet_b0_results.json"),
    ("mobilenetv3_large", "MobileNetV3-Large", "mobilenetv3_large_results.json"),
]


def load_accuracy(json_path: Path) -> float | None:
    """Read accuracy from a results JSON.

    K-fold JSONs (when present) take precedence — they are the more reliable
    source. Returns the first stable fold's accuracy (or aggregated.mean if
    only aggregated is present).
    """
    try:
        data = json.loads(json_path.read_text())
    except FileNotFoundError:
        return None
    # Aggregated K-fold mean is the most reliable number.
    if "aggregated" in data and isinstance(data["aggregated"], dict):
        mean = data["aggregated"].get("accuracy_mean")
        if mean is not None:
            return float(mean) * 100
    # Single-fold accuracy.
    acc = data.get("accuracy")
    if acc is None:
        return None
    return float(acc) * 100  # convert to percentage


def find_headline_acc_pcts(text: str, label: str) -> list[str]:
    """Find acc percentages in table rows that contain this label.

    Scans markdown table rows only (lines starting with '|' and containing '|'),
    looking for either `**<label>**` (bold label) or `<label>` followed by a
    parenthesized detail (e.g. 'Self-built CNN (4 conv blocks ...)') in the
    first cell. Then extracts all bold `**NN.NN%**` percentages from the
    same row.

    Returns a list of matched accuracy strings (preserves row order).
    """
    found: list[str] = []
    # Either bold label, or label starting followed by '(' detail
    label_re_bold = re.compile(rf"\*\*{re.escape(label)}\*\*", re.IGNORECASE)
    label_re_paren = re.compile(rf"{re.escape(label)}\s*\(", re.IGNORECASE)
    acc_re = re.compile(r"\*\*(\d{2,3}\.\d{2}%)\*\*")
    for line in text.splitlines():
        if "|" not in line or not line.lstrip().startswith("|"):
            continue
        if not (label_re_bold.search(line) or label_re_paren.search(line)):
            continue
        for m in acc_re.finditer(line):
            found.append(m.group(1))
    return found


def main() -> int:
    if not README.exists():
        print(f"ERROR: {README} not found", file=sys.stderr)
        return 2

    readme_text = README.read_text()
    print(f"Checking {README.relative_to(PROJECT_ROOT)} "
          f"against {RESULTS_DIR.relative_to(PROJECT_ROOT)}/*_results.json")
    print()

    errors = []
    for model, label, json_name in MODELS:
        json_path = RESULTS_DIR / json_name
        canonical = load_accuracy(json_path)
        if canonical is None:
            print(f"  [{model:>10s}] SKIP — {json_name} not found or has no 'accuracy' key")
            continue

        headlines = find_headline_acc_pcts(readme_text, label)
        if not headlines:
            print(f"  [{model:>10s}] WARNING — no headline percentages found for '{label}'")
            errors.append(model)
            continue

        # Compare each headline percentage to canonical. Tolerate ±0.05pp (rounding).
        canonical_str = f"{canonical:.2f}"
        for headline in headlines:
            clean = re.sub(r"\*", "", headline)
            if clean == canonical_str:
                print(f"  [{model:>10s}] OK    — {label} = {clean} matches JSON")
            else:
                # Allow close-match (rounding)
                try:
                    if abs(float(clean.rstrip("%")) - canonical) < 0.05:
                        print(f"  [{model:>10s}] OK    — {label} = {clean} ≈ JSON {canonical_str} (rounding)")
                        continue
                except ValueError:
                    pass
                print(f"  [{model:>10s}] DRIFT — {label} headline {clean} != JSON {canonical_str}")
                errors.append(model)

    print()
    if errors:
        print(f"FAIL: {len(errors)} drift(s) detected: {sorted(set(errors))}")
        print("Fix README.md headline so each model label matches its results JSON.")
        return 1
    print("PASS: README headline is consistent with all results JSONs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())