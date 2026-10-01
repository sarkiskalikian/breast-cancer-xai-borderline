"""Recall-oriented evaluation helpers shared by the notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score
from sklearn.utils import resample


def threshold_sweep(scores: np.ndarray, y_true: np.ndarray,
                    thresholds: np.ndarray = np.arange(-1, 1, 0.05)) -> pd.DataFrame:
    """Recall / precision / F1 at each decision threshold (used for the SVM decision function)."""
    rows = []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        rows.append({"threshold": round(float(t), 2),
                     "recall": recall_score(y_true, y_pred, zero_division=0),
                     "precision": precision_score(y_true, y_pred, zero_division=0),
                     "f1": f1_score(y_true, y_pred, zero_division=0)})
    return pd.DataFrame(rows)


def tune_threshold_f2(y_true: np.ndarray, y_proba: np.ndarray,
                      thresholds: np.ndarray = np.linspace(0.1, 0.9, 17)) -> dict:
    """Pick the probability cutoff that maximises F2 on a validation set.

    F2 weights recall twice as much as precision, matching the clinical goal of
    missing as few cancer cases as possible.
    """
    best = {"thr": 0.5, "f2": -1.0}
    for t in thresholds:
        f2 = fbeta_score(y_true, (y_proba >= t).astype(int), beta=2, zero_division=0)
        if f2 > best["f2"]:
            best = {"thr": float(t), "f2": float(f2)}
    return best


def binary_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


def downsample_majority(X: pd.DataFrame, y: pd.Series, majority: int = 0,
                        random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Downsample the majority class to the combined size of the minority classes."""
    y = pd.Series(np.asarray(y), index=X.index)
    is_major = y == majority
    n_keep = int((~is_major).sum())
    kept = resample(X.index[is_major], replace=False, n_samples=n_keep, random_state=random_state)
    idx = np.concatenate([kept, X.index[~is_major]])
    return X.loc[idx], y.loc[idx]
