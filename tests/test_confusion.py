"""
Tests for src.confusion — confusion matrix, per-class metrics, ROC + AUC.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.confusion import (  # noqa: E402
    confusion_matrix,
    needs_scores_for_roc,
    per_class_metrics_from_cm,
    roc_curve,
)


def test_confusion_matrix_perfect_classifier():
    """A perfect classifier produces a diagonal confusion matrix."""
    labels = [0, 0, 1, 1, 0, 1]
    preds = [0, 0, 1, 1, 0, 1]
    cm = confusion_matrix(labels, preds, num_classes=2)
    assert cm.shape == (2, 2)
    assert np.array_equal(cm, np.array([[3, 0], [0, 3]]))


def test_confusion_matrix_off_diagonal():
    """Off-diagonal entries reflect misclassifications (rows=true, cols=pred)."""
    labels = [0, 0, 1, 1]
    preds = [0, 1, 0, 1]  # 1 false positive, 1 false negative
    cm = confusion_matrix(labels, preds, num_classes=2)
    assert cm[0, 0] == 1  # TN
    assert cm[0, 1] == 1  # FP (true NoCrack → predicted Crack)
    assert cm[1, 0] == 1  # FN (true Crack → predicted NoCrack)
    assert cm[1, 1] == 1  # TP


def test_confusion_matrix_class_index_map():
    """Index 0 = NoCrack, index 1 = Crack. Rows = true label."""
    labels = [1, 1, 1, 0]
    preds = [1, 1, 0, 1]
    cm = confusion_matrix(labels, preds, num_classes=2)
    # Crack rows = first 3 entries, col[0] = correctly predicted NoCrack
    assert cm[0, 0] == 0  # NoCrack correctly predicted as NoCrack
    assert cm[1, 1] == 2  # Crack correctly predicted as Crack
    assert cm[1, 0] == 1  # Crack predicted as NoCrack (FN)
    assert cm[0, 1] == 1  # NoCrack predicted as Crack (FP)


def test_per_class_metrics_hand_calculated():
    """Per-class metrics from a hand-derivable confusion matrix.

    cm is rows=true / cols=pred, so for the NoCrack class (idx 0):
        TP = cm[0, 0]
        FP = sum(col 0) - TP   (other rows in col 0 are FPs)
        FN = sum(row 0) - TP   (other cols in row 0 are FNs)
    """
    cm = np.array([
        [50, 10],   # NoCrack: 50 correctly predicted NoCrack, 10 misclassified as Crack
        [5,  35],   # Crack:   5 misclassified as NoCrack, 35 correctly predicted Crack
    ])
    m = per_class_metrics_from_cm(cm)
    # NoCrack: P = TP / (TP + FP) = 50 / (50 + 5) = 50/55
    np.testing.assert_allclose(m["NoCrack_precision"], 50 / 55, rtol=1e-6)
    # NoCrack: R = TP / (TP + FN) = 50 / (50 + 10) = 50/60
    np.testing.assert_allclose(m["NoCrack_recall"], 50 / 60, rtol=1e-6)
    # Crack: P = 35 / (35 + 10) = 35/45
    np.testing.assert_allclose(m["Crack_precision"], 35 / 45, rtol=1e-6)
    # Crack: R = 35 / (35 + 5) = 35/40
    np.testing.assert_allclose(m["Crack_recall"], 35 / 40, rtol=1e-6)
    # Crack F1 = 2 * (35/45 * 35/40) / (35/45 + 35/40)
    p_c, r_c = 35 / 45, 35 / 40
    np.testing.assert_allclose(m["Crack_f1"], 2 * p_c * r_c / (p_c + r_c), rtol=1e-6)
    assert m["NoCrack_support"] == 60
    assert m["Crack_support"] == 40


def test_per_class_metrics_zero_denominator_safety():
    """If a class has zero predictions, return 0 instead of NaN."""
    cm = np.array([[10, 0], [0, 0]])  # Only NoCrack samples, all predicted correctly
    m = per_class_metrics_from_cm(cm)
    assert m["Crack_precision"] == 0.0
    assert m["Crack_recall"] == 0.0
    assert m["Crack_f1"] == 0.0


def test_roc_curve_perfect_separation_auc_one():
    """Perfect ranking of positives above negatives → AUC = 1.0."""
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    fpr, tpr, thresholds, auc = roc_curve(labels, scores)
    assert auc == pytest.approx(1.0, abs=1e-6)
    assert len(fpr) == len(tpr)
    assert len(fpr) == len(thresholds)


def test_roc_curve_random_auc_half():
    """Random ordering → AUC ≈ 0.5."""
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    scores = [0.5, 0.4, 0.6, 0.3, 0.55, 0.45, 0.35, 0.65]
    _fpr, _tpr, _thr, auc = roc_curve(labels, scores)
    assert auc == pytest.approx(0.5, abs=0.05)


def test_roc_curve_inverse_separation_auc_zero():
    """All positives ranked below all negatives → AUC = 0.0."""
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    scores = [0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4]
    _fpr, _tpr, _thr, auc = roc_curve(labels, scores)
    assert auc == pytest.approx(0.0, abs=1e-6)


def test_roc_curve_length_matches_thresholds():
    """Number of points should equal the requested num_thresholds."""
    labels = [0, 1] * 5
    scores = np.linspace(0, 1, 10)
    fpr, tpr, thr, _ = roc_curve(labels, scores, num_thresholds=51)
    assert len(fpr) == 51
    assert len(tpr) == 51
    assert len(thr) == 51


def test_needs_scores_for_roc_detects_missing_key():
    """If results JSON has only hard predictions, the demo should know ROC is unavailable."""
    d = {"accuracy": 0.86, "predictions": [0, 1], "labels": [0, 1]}
    assert needs_scores_for_roc(d) is True


def test_needs_scores_for_roc_detects_present_key():
    """If results JSON has a 'scores' array, ROC can be drawn."""
    d = {"scores": [0.1, 0.9], "labels": [0, 1]}
    assert needs_scores_for_roc(d) is False
