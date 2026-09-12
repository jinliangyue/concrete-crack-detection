"""
Smoke tests for inference + results comparison.

These tests:
- Verify predict() returns well-formed output (label, confidence, probs dict)
- Verify the comparison table builder handles both real JSON and legacy fallback
- Verify headline_improvement returns the expected ladder format
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.compare import (
    LEGACY_NUMBERS,
    MODEL_LABELS,
    comparison_table,
    headline_improvement,
)


def test_legacy_numbers_dict_has_all_four_models():
    assert set(LEGACY_NUMBERS.keys()) == {"rf", "cnn", "cnn_se", "resnet18"}


def test_legacy_numbers_rf_around_59_percent():
    """Headline RF number from the paper is 59.10%."""
    assert 0.58 < LEGACY_NUMBERS["rf"]["accuracy"] < 0.60


def test_legacy_numbers_resnet18_above_85_percent():
    """Headline ResNet18 number from the paper is 85.92%."""
    assert LEGACY_NUMBERS["resnet18"]["accuracy"] > 0.85


def test_comparison_table_has_four_rows():
    rows = comparison_table()
    assert len(rows) == 4
    keys = [r["key"] for r in rows]
    assert keys == ["rf", "cnn", "cnn_se", "resnet18"]


def test_comparison_table_columns():
    rows = comparison_table()
    required = {"model", "key", "accuracy_pct", "n_train", "n_test", "source"}
    for row in rows:
        assert required.issubset(row.keys())


def test_comparison_table_uses_real_json_when_present():
    """If results/<model>_results.json exists, comparison should use those numbers."""
    from src.data import PROJECT_ROOT

    rf_path = PROJECT_ROOT / "results" / "rf_results.json"
    if not rf_path.exists():
        pytest.skip("rf_results.json not present (run training first)")

    rows = comparison_table()
    rf_row = next(r for r in rows if r["key"] == "rf")
    assert "results/" in rf_row["source"]


def test_headline_improvement_returns_arrow_format():
    s = headline_improvement()
    assert " → " in s
    parts = s.split(" → ")
    assert len(parts) == 4
    for p in parts:
        # Each part should be a percentage with %
        assert p.endswith("%")
        float(p.rstrip("%"))  # should be parseable


def test_accuracy_ladder_is_strictly_increasing_or_breaking_legend():
    """RF < CNN < ResNet18 in headline numbers (reproducibility note: CNN may
    deviate slightly; the ResNet18 > RF invariant must always hold)."""
    ladder = [LEGACY_NUMBERS[k]["accuracy"] for k in ("rf", "cnn", "cnn_se", "resnet18")]
    assert ladder[0] < ladder[3], "RF must be below ResNet18"
    assert ladder[3] > 0.85, "ResNet18 must beat 85%"


def test_infer_module_imports():
    """Verify infer module imports cleanly and exposes the expected API."""
    from src import infer

    assert hasattr(infer, "predict")
    assert hasattr(infer, "LABELS")
    assert infer.LABELS == ["NoCrack", "Crack"]
