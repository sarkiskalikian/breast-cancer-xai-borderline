"""Loading and cleaning the BCSC risk-factor data.

This is the preprocessing every notebook applies before modelling:

1. concatenate the three BCSC files and drop ``year`` and ``count``;
2. treat code 9 ("unknown") as missing in every column except
   ``age_group_5_years`` (where 9 is a real age band);
3. drop rows with an unknown target, fill the remaining gaps with each
   column's mode;
4. one-hot encode ``race_eth`` (first level dropped).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "breast_cancer_history"
BCSC_FILES = [f"bcsc_risk_factors_summarized{i}_092020.csv" for i in (1, 2, 3)]


def load_bcsc(data_dir: str | Path = "data") -> pd.DataFrame:
    data_dir = Path(data_dir)
    return pd.concat([pd.read_csv(data_dir / name) for name in BCSC_FILES], ignore_index=True)


def preprocess(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    df = df.drop(columns=["year", "count"], errors="ignore").copy()

    others = df.columns[df.columns != "age_group_5_years"]
    df[others] = df[others].replace(9, np.nan)
    df = df.dropna(subset=[target])

    for col in df.columns:
        if col != target and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    df = pd.get_dummies(df, columns=["race_eth"], drop_first=True)
    bool_cols = df.select_dtypes("bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    df[target] = df[target].astype(int)
    return df.reset_index(drop=True)


def features_and_target(df: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target]), df[target]
