"""The borderline (third) class.

A standard binary model treats a healthy patient with a predicted cancer
probability of 0.49 the same as one at 0.05. We separate them: a logistic
regression is fit on the binary label, and patients **without** cancer whose
predicted probability lies in [0.35, 0.50] become class 2, "borderline".
They look like cancer cases on these risk factors and are worth a closer look.

The 0.50 upper bound keeps anyone the baseline would already call "cancer"
out of the group. For the lower bound we tried 0.30 (too wide: many clearly
negative cases) and 0.40 (too small to learn from); 0.35 kept the group
meaningful.

Labels: 0 = no cancer, 1 = cancer, 2 = borderline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

LOWER, UPPER = 0.35, 0.50


def label_borderline(y: np.ndarray | pd.Series, proba: np.ndarray,
                     lower: float = LOWER, upper: float = UPPER) -> np.ndarray:
    """Turn binary labels into three classes given each patient's cancer probability."""
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba)
    labels = y.copy()
    labels[(y == 0) & (proba >= lower) & (proba <= upper)] = 2
    return labels


def add_borderline_class(X: pd.DataFrame, y: pd.Series, lower: float = LOWER, upper: float = UPPER,
                         max_iter: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """Fit the logistic-regression baseline on all data and return (three-class labels, probabilities)."""
    baseline = LogisticRegression(max_iter=max_iter)
    baseline.fit(X, y)
    proba = baseline.predict_proba(X)[:, 1]
    return label_borderline(y, proba, lower, upper), proba
