"""Load, validate, clean and split the churn data.

Produces three files in data/processed/: train.csv, val.csv, test.csv
(60 / 20 / 20, stratified on the target).  The test set is only used once,
in src/evaluate.py.

Run:  python -m src.data_prep
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_raw(path=config.RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.generate_data` or place the "
            "Telco churn CSV there."
        )
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame, need_target: bool = True) -> None:
    """Raise ValueError with a clear message if the data is unusable."""
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if need_target:
        if config.TARGET not in df.columns:
            raise ValueError(f"Target column '{config.TARGET}' not found")
        bad = set(df[config.TARGET].dropna().unique()) - {0, 1}
        if bad or df[config.TARGET].isna().any():
            raise ValueError(f"Target must be binary 0/1 with no nulls, got {bad or 'nulls'}")
    if (pd.to_numeric(df["tenure"], errors="coerce") < 0).any():
        raise ValueError("tenure contains negative values")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types, encode the target and remove duplicates."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    # In the public file TotalCharges holds blank strings for tenure-0 rows.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip().replace("", np.nan),
        errors="coerce",
    )
    df[config.TARGET] = df[config.TARGET].map({"Yes": 1, "No": 0}).fillna(df[config.TARGET])
    if config.ID_COL in df.columns:
        df = df.drop_duplicates(subset=config.ID_COL)
        df = df.drop(columns=config.ID_COL)
    df = df.drop_duplicates().reset_index(drop=True)
    validate_schema(df)
    df[config.TARGET] = df[config.TARGET].astype(int)
    return df


def split(df: pd.DataFrame, seed: int = config.SEED):
    """Stratified train / validation / test split (60/20/20)."""
    train_val, test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df[config.TARGET], random_state=seed
    )
    val_fraction = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_fraction, stratify=train_val[config.TARGET],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_split(name: str) -> pd.DataFrame:
    path = config.PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m src.data_prep` first.")
    return pd.read_csv(path)


def xy(df: pd.DataFrame):
    return df.drop(columns=[config.TARGET]), df[config.TARGET]


def main() -> None:
    raw = load_raw()
    print(f"Raw data: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    df = clean(raw)
    print(f"After cleaning: {df.shape[0]:,} rows, churn rate {df[config.TARGET].mean():.1%}")
    print(f"Missing values: {df.isna().sum()[df.isna().sum() > 0].to_dict()}")

    train, val, test = split(df)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(config.PROCESSED_DIR / f"{name}.csv", index=False)
        print(f"  {name:5s}: {len(part):5,} rows, churn rate {part[config.TARGET].mean():.1%}")

    # a small file of unlabeled rows to demo src/predict.py
    sample = test.drop(columns=[config.TARGET]).head(20)
    sample.to_csv(config.PROCESSED_DIR / "sample_new_customers.csv", index=False)
    print(f"Saved splits to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
