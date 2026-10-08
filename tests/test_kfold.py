"""
Tests for K-fold cross-validation utilities.

These cover the math/contract: fold sizes, stratification, determinism.
The end-to-end training loop is covered separately in test_pipeline_e2e.py.
"""

import numpy as np
import pytest

from src.kfold import aggregate_folds, make_kfold_indices, per_fold_metrics


def _make_balanced_y(n_crack: int = 50, n_nocrack: int = 50) -> np.ndarray:
    return np.array([1] * n_crack + [0] * n_nocrack)


def test_kfold_returns_n_folds_pairs():
    y = _make_balanced_y()
    folds = make_kfold_indices(y, n_folds=5)
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)


def test_kfold_indices_disjoint():
    y = _make_balanced_y()
    folds = make_kfold_indices(y, n_folds=5)
    all_test = np.concatenate([test_idx for _, test_idx in folds])
    # Each index appears exactly once across all folds.
    assert len(all_test) == len(y)
    assert len(np.unique(all_test)) == len(y)


def test_kfold_stratified_class_ratio_preserved():
    y = _make_balanced_y(n_crack=80, n_nocrack=20)  # 4:1 imbalanced
    folds = make_kfold_indices(y, n_folds=5)
    for train_idx, test_idx in folds:
        # Each fold's test slice should have ~80/20 split (within rounding).
        train_pos = int(y[train_idx].sum())
        test_pos = int(y[test_idx].sum())
        test_total = len(test_idx)
        ratio = test_pos / test_total
        assert 0.65 < ratio < 0.95, (
            f"Test fold ratio {ratio:.2f} outside [0.65, 0.95]; "
            f"stratification broken (test_pos={test_pos}, total={test_total})"
        )


def test_kfold_seed_reproducibility():
    y = _make_balanced_y(n_crack=30, n_nocrack=70)
    a = make_kfold_indices(y, n_folds=5, seed=42)
    b = make_kfold_indices(y, n_folds=5, seed=42)
    for (at, ae), (bt, be) in zip(a, b):
        np.testing.assert_array_equal(at, bt)
        np.testing.assert_array_equal(ae, be)


def test_kfold_different_seed_changes_split():
    y = _make_balanced_y(n_crack=50, n_nocrack=50)
    a = make_kfold_indices(y, n_folds=5, seed=42)
    b = make_kfold_indices(y, n_folds=5, seed=123)
    # At least one fold's test slice should differ.
    differ = any(not np.array_equal(ae, be) for (_, ae), (_, be) in zip(a, b))
    assert differ, "Different seeds produced identical splits — seed is not honored"


def test_kfold_total_samples_equals_y():
    y = _make_balanced_y()
    folds = make_kfold_indices(y, n_folds=5)
    for train_idx, test_idx in folds:
        assert len(train_idx) + len(test_idx) == len(y)


def test_per_fold_metrics_extracts_six_keys():
    """per_fold_metrics should pull accuracy + precision + recall + f1 + fpr + fnr.

    When predictions + labels are passed, FPR/FNR are derived from the
    confusion matrix. Without them, FPR falls back to 0.0 and FNR to
    1 - recall (which is exact for balanced splits).
    """
    fake_report = {
        "accuracy": 0.87,
        "Crack": {
            "precision": 0.92,
            "recall": 0.81,
            "f1-score": 0.86,
            "support": 600,
        },
        "NoCrack": {
            "precision": 0.83,
            "recall": 0.93,
            "f1-score": 0.88,
            "support": 600,
        },
    }
    m = per_fold_metrics(fake_report)
    assert m["accuracy"] == 0.87
    assert m["precision"] == 0.92
    assert m["recall"] == 0.81
    assert m["f1"] == 0.86
    assert m["fpr"] == 0.0  # no predictions/labels → fallback
    assert abs(m["fnr"] - (1.0 - 0.81)) < 1e-9


def test_per_fold_metrics_derives_fpr_from_confusion_matrix():
    """When predictions + labels are passed, FPR must come from the confusion matrix.

    A perfect classifier on a balanced test set should give fpr=0 and fnr=0.
    A worst-case classifier (predicts all Crack) on the same data should give
    fpr=1 and fnr=0.
    """
    fake_report = {
        "accuracy": 0.5,
        "Crack": {"precision": 0.5, "recall": 1.0, "f1-score": 0.667, "support": 100},
        "NoCrack": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 100},
    }
    # All-correct predictions → fpr=0, fnr=0.
    preds_all_correct = [0] * 100 + [1] * 100
    labels_balanced = [0] * 100 + [1] * 100
    m = per_fold_metrics(fake_report, predictions=preds_all_correct, labels=labels_balanced)
    assert m["fpr"] == 0.0
    assert m["fnr"] == 0.0

    # All-wrong (predict all Crack) → fpr=1.0, fnr=0.
    preds_all_crack = [1] * 200
    m = per_fold_metrics(fake_report, predictions=preds_all_crack, labels=labels_balanced)
    assert m["fpr"] == 1.0
    assert m["fnr"] == 0.0  # all positives predicted as positives → 0% miss

    # Half-miss on positives → fpr=0, fnr=0.5
    half_miss_preds = [0] * 100 + [1] * 50 + [0] * 50
    m = per_fold_metrics(fake_report, predictions=half_miss_preds, labels=labels_balanced)
    assert m["fpr"] == 0.0
    assert m["fnr"] == 0.5


def test_aggregate_folds_returns_mean_std_min_max():
    folds = [
        {"accuracy": 0.85, "precision": 0.90, "recall": 0.80, "f1": 0.85, "fpr": 0.10, "fnr": 0.20},
        {"accuracy": 0.87, "precision": 0.91, "recall": 0.82, "f1": 0.86, "fpr": 0.09, "fnr": 0.18},
        {"accuracy": 0.89, "precision": 0.93, "recall": 0.84, "f1": 0.88, "fpr": 0.08, "fnr": 0.16},
        {"accuracy": 0.86, "precision": 0.92, "recall": 0.81, "f1": 0.86, "fpr": 0.09, "fnr": 0.19},
        {"accuracy": 0.88, "precision": 0.91, "recall": 0.83, "f1": 0.87, "fpr": 0.08, "fnr": 0.17},
    ]
    agg = aggregate_folds(folds)
    assert agg["n_folds"] == 5
    # mean ≈ 0.87 ± 0.015
    assert abs(agg["accuracy_mean"] - 0.87) < 1e-9
    assert agg["accuracy_std"] > 0
    assert agg["accuracy_min"] == 0.85
    assert agg["accuracy_max"] == 0.89


def test_aggregate_folds_handles_empty_input():
    agg = aggregate_folds([])
    assert agg == {"n_folds": 0}