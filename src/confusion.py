"""
Confusion-matrix + ROC-curve utilities for crack detection models.

Pure-Python / NumPy implementations so the demo can render them from the
predictions array stored in results/<model>_results.json — no extra deps.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def confusion_matrix(
    labels: Sequence[int],
    predictions: Sequence[int],
    num_classes: int = 2,
) -> np.ndarray:
    """Return a (num_classes, num_classes) confusion matrix.

    Rows = true label, cols = predicted label. Index 0 = NoCrack, 1 = Crack.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_true, y_pred in zip(labels, predictions):
        cm[int(y_true), int(y_pred)] += 1
    return cm


def per_class_metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    """Derive precision / recall / F1 / support from a 2-class confusion matrix."""
    out: Dict[str, float] = {}
    for idx, name in enumerate(("NoCrack", "Crack")):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - tp)
        fn = float(cm[idx, :].sum() - tp)
        support = int(cm[idx, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[f"{name}_precision"] = precision
        out[f"{name}_recall"] = recall
        out[f"{name}_f1"] = f1
        out[f"{name}_support"] = support
    return out


def roc_curve(
    labels: Sequence[int],
    scores: Sequence[float],
    num_thresholds: int = 101,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute the ROC curve + AUC for a binary classifier.

    Parameters
    ----------
    labels
        Ground-truth binary labels (0 = NoCrack, 1 = Crack).
    scores
        Predicted probability of class 1 (Crack) for each sample.
    num_thresholds
        Number of evenly-spaced thresholds in [0, 1]. Default 101.

    Returns
    -------
    fpr, tpr, thresholds, auc
        fpr/tpr arrays of length ``num_thresholds`` (or fewer if ties collapse
        thresholds); thresholds array of the same length. AUC via the
        trapezoidal rule.
    """
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    assert y.shape == s.shape, "labels and scores must have the same length"

    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    fpr_list: List[float] = []
    tpr_list: List[float] = []

    pos = float((y == 1).sum())
    neg = float((y == 0).sum())

    for thr in thresholds:
        y_pred = (s >= thr).astype(np.int64)
        tp = float(((y == 1) & (y_pred == 1)).sum())
        fp = float(((y == 0) & (y_pred == 1)).sum())
        tpr = tp / pos if pos else 0.0
        fpr = fp / neg if neg else 0.0
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    fpr = np.asarray(fpr_list)
    tpr = np.asarray(tpr_list)

    # Trapezoidal AUC. Sort by FPR so non-monotonic inputs still integrate correctly.
    order = np.argsort(fpr)
    fpr_s = fpr[order]
    tpr_s = tpr[order]
    auc = float(np.trapz(tpr_s, fpr_s))
    return fpr, tpr, thresholds, auc


def needs_scores_for_roc(results_json: Dict) -> bool:
    """Heuristic: predictions array is enough for confusion matrix, but ROC
    needs continuous scores. We currently store hard predictions only, so
    the demo's ROC falls back to a "no scores available" message unless we
    also save the positive-class probabilities."""
    return "scores" not in results_json
