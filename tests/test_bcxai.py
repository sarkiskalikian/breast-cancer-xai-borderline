import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bcxai.borderline import add_borderline_class, label_borderline  # noqa: E402
from bcxai.data import preprocess  # noqa: E402
from bcxai.evaluation import downsample_majority, tune_threshold_f2  # noqa: E402


def toy_bcsc():
    """A few rows in the BCSC coding, where 9 means unknown."""
    return pd.DataFrame({
        "year": [2005] * 6,
        "age_group_5_years": [9, 3, 4, 9, 5, 6],     # 9 is a real age band here
        "race_eth": [1, 2, 1, 9, 3, 1],
        "BIRADS_breast_density": [2, 9, 3, 2, 2, 4],
        "biophx": [0, 1, 9, 0, 1, 0],
        "breast_cancer_history": [0, 1, 0, 9, 1, 0],
        "count": [10, 3, 5, 1, 2, 8],
    })


def test_preprocess_handles_unknown_codes():
    df = preprocess(toy_bcsc())
    assert "year" not in df and "count" not in df
    assert len(df) == 5                                   # row with unknown target dropped
    assert 9 in df["age_group_5_years"].values            # kept as an age band
    assert df["BIRADS_breast_density"].isin([2, 3, 4]).all()   # 9 replaced by the mode
    assert any(c.startswith("race_eth_") for c in df.columns)
    assert df.isna().sum().sum() == 0


def test_borderline_only_relabels_healthy_patients_in_the_band():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.10, 0.35, 0.42, 0.51, 0.40, 0.90])
    assert label_borderline(y, p).tolist() == [0, 2, 2, 0, 1, 1]


def test_add_borderline_class_returns_three_labels():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] + 0.5 * rng.normal(size=400) > 0.3).astype(int))
    labels, proba = add_borderline_class(X, y)
    assert set(np.unique(labels)) <= {0, 1, 2}
    assert ((labels == 2) <= (y.values == 0)).all()      # cancer cases are never moved
    assert ((proba[labels == 2] >= 0.35) & (proba[labels == 2] <= 0.5)).all()


def test_threshold_tuning_prefers_recall():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.2, 0.3, 0.45, 0.42, 0.9])
    assert tune_threshold_f2(y, p)["thr"] <= 0.45


def test_downsampling_balances_majority():
    X = pd.DataFrame({"x": range(20)})
    y = pd.Series([0] * 14 + [1] * 4 + [2] * 2)
    Xd, yd = downsample_majority(X, y)
    assert (yd == 0).sum() == 6 and len(Xd) == len(yd) == 12
