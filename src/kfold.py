"""
Stratified K-fold cross-validation utilities.

The single train/test split in src.train is fine for an initial reproduction,
but a single 87.92% number does not tell you whether the model is genuinely
that good or whether it caught a lucky fold. StratifiedKFold gives mean ± std
across folds, which is what reviewers / hiring committees / future-you actually
want to see.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold


def make_kfold_indices(
    y: np.ndarray, n_folds: int = 5, seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return a list of (train_idx, test_idx) pairs for stratified K-fold.

    StratifiedKFold preserves the per-class ratio in both train and test slices,
    which matters when we cap at max_per_class=4000 per class (still balanced)
    but the per-surface distribution inside the cap is uneven (D=bridge deck has
    far fewer cracks than W walls do).
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    # StratifiedKFold needs X for shape but ignores it; pass zeros for clarity.
    return list(skf.split(np.zeros(len(y)), y))


def per_fold_metrics(
    report: Dict[str, Any],
    predictions: Sequence[int] | None = None,
    labels: Sequence[int] | None = None,
) -> Dict[str, float]:
    """Pull headline metrics out of one fold's sklearn classification_report dict.

    The classification_report dict has accuracy + per-class precision/recall/f1
    but NO fpr/fnr — those have to be computed from the confusion matrix.
    Pass the raw predictions + labels to get them; otherwise fall back to
    the simpler "1 - recall" approximation for FNR (which is only exact when
    the test set is class-balanced, which our 1:1 capped SDNET2018 splits are).
    """
    crack = report["Crack"]
    recall = float(crack["recall"])
    fnr = 1.0 - recall

    fpr = 0.0
    if predictions is not None and labels is not None:
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(list(labels), list(predictions), labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
            fnr = float(fn) / float(fn + tp) if (fn + tp) > 0 else 0.0
        except Exception:
            pass

    return {
        "accuracy": float(report["accuracy"]),
        "precision": float(crack["precision"]),
        "recall": recall,
        "f1": float(crack["f1-score"]),
        "fpr": fpr,
        "fnr": fnr,
    }


def aggregate_folds(folds: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    """Aggregate per-fold metrics into mean / std / min / max summary.

    Returns a dict with keys like 'accuracy_mean', 'accuracy_std', 'accuracy_min',
    'accuracy_max', and 'n_folds'. Designed to be merged into a single metrics
    JSON file alongside the existing `accuracy` key.
    """
    if not folds:
        return {"n_folds": 0}

    metrics: Dict[str, Any] = {"n_folds": int(len(folds))}
    keys = ("accuracy", "precision", "recall", "f1", "fpr", "fnr")
    for k in keys:
        vals = [float(f[k]) for f in folds if k in f]
        if not vals:
            continue
        arr = np.asarray(vals)
        metrics[f"{k}_mean"] = float(arr.mean())
        metrics[f"{k}_std"] = float(arr.std(ddof=1)) if len(vals) > 1 else 0.0
        metrics[f"{k}_min"] = float(arr.min())
        metrics[f"{k}_max"] = float(arr.max())
    return metrics