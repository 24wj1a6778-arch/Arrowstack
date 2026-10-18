"""Feature engineering + preprocessing.

`build_pipeline(estimator)` returns ONE sklearn Pipeline that accepts the
cleaned raw columns, so the exact same object is used in training, evaluation
and prediction (no training/serving skew).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config

TENURE_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create business-meaningful features from raw columns."""
    X = X.copy()
    tenure = pd.to_numeric(X["tenure"], errors="coerce")

    X["tenure_group"] = pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    X["avg_monthly_spend"] = X["TotalCharges"] / tenure.replace(0, np.nan)
    X["num_services"] = sum((X[c] == "Yes").astype(int) for c in config.SERVICE_COLUMNS)
    X["charge_per_service"] = X["MonthlyCharges"] / (X["num_services"] + 1)
    X["is_month_to_month"] = (X["Contract"] == "Month-to-month").astype(int)
    X["autopay"] = X["PaymentMethod"].astype(str).str.contains("automatic", case=False).astype(int)
    return X


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
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
