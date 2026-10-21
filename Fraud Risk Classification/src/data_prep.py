"""Load, validate, clean and split the transaction data.

Produces train.csv / val.csv / test.csv (60/20/20, stratified on the target)
in data/processed/.  The test set is only used once, in src/evaluate.py.

Run:  python -m src.data_prep
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_raw(path=config.RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.generate_data` first.")
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame, need_target: bool = True) -> None:
    """Raise ValueError with a clear message if the data is unusable."""
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if need_target:
        if config.TARGET not in df.columns:
            raise ValueError(f"Target column '{config.TARGET}' not found")
        if df[config.TARGET].isna().any() or not set(df[config.TARGET].unique()) <= {0, 1}:
            raise ValueError("Target must be binary 0/1 with no nulls")
    amount = pd.to_numeric(df[config.AMOUNT_COL], errors="coerce")
    if (amount < 0).any():
        raise ValueError("amount contains negative values")
    hour = pd.to_numeric(df["hour"], errors="coerce")
    if ((hour < 0) | (hour > 23)).any():
        raise ValueError("hour must be between 0 and 23")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types, drop duplicates and the ID column, validate."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    if config.ID_COL in df.columns:
        df = df.drop_duplicates(subset=config.ID_COL).drop(columns=config.ID_COL)
    df = df.drop_duplicates().reset_index(drop=True)
    for col in config.RAW_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in config.RAW_CATEGORICAL:
        df[col] = df[col].astype("object").where(df[col].notna(), None)
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) and v.strip() else None)
    validate_schema(df)
    df[config.TARGET] = df[config.TARGET].astype(int)
    return df


def split(df: pd.DataFrame, seed: int = config.SEED):
    """Stratified train / validation / test split (60/20/20)."""
    train_val, test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df[config.TARGET], random_state=seed)
    val_fraction = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_fraction, stratify=train_val[config.TARGET],
        random_state=seed)
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
    miss = df.isna().sum()
    print(f"After cleaning: {len(df):,} rows, fraud rate {df[config.TARGET].mean():.2%} "
          f"({int(df[config.TARGET].sum())} frauds)")
    print(f"Missing values: {miss[miss > 0].to_dict()}")

    train, val, test = split(df)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(config.PROCESSED_DIR / f"{name}.csv", index=False)
        print(f"  {name:5s}: {len(part):6,} rows, {int(part[config.TARGET].sum()):4d} frauds "
              f"({part[config.TARGET].mean():.2%})")

    # unlabeled sample for the predict.py demo: all test frauds up to 8, plus legit rows
    sample = pd.concat([test[test[config.TARGET] == 1].head(8),
                        test[test[config.TARGET] == 0].head(22)]).sample(frac=1, random_state=config.SEED)
    sample.drop(columns=[config.TARGET]).to_csv(
        config.PROCESSED_DIR / "sample_new_transactions.csv", index=False)
    print(f"Saved splits to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
