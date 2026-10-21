"""Feature engineering + preprocessing.

`build_pipeline(estimator)` returns ONE sklearn Pipeline that accepts the
cleaned raw columns, so the same object is used in training, evaluation and
prediction (no training/serving skew).  Resampling (if any) happens INSIDE the
final step and only during fit, so validation/test data are never resampled.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create fraud-relevant features from raw columns."""
    X = X.copy()
    num = {c: pd.to_numeric(X[c], errors="coerce") for c in config.RAW_NUMERIC}
    X["amount_ratio"] = num["amount"] / (num["avg_amount_30d"] + 1)       # vs customer's norm
    X["log_amount"] = np.log1p(num["amount"].clip(lower=0))
    X["is_night"] = (num["hour"] <= 5).astype(int)
    X["intl_online"] = ((num["is_international"] == 1) & (X["channel"] == "online")).astype(int)
    X["new_card"] = (num["card_age_days"] < 60).astype(int)
    return X


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", build_preprocessor()),
        ("model", estimator),
    ])
