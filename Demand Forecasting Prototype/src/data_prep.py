"""Load, validate, clean and split the sales data - IN TIME ORDER.

Time series must never be split randomly: the model would see the future.
Layout (relative to the last date in the data):

    |<------------- train ------------->|<-- validation 84d -->|<-- test 84d -->|

Outputs in data/processed/:
  demand_daily.csv    clean daily panel (complete calendar per product)
  split_info.json     the split dates
  sample_inventory.csv, sample_future_promos.csv   inputs for the predict.py demo

Run:  python -m src.data_prep
"""
import json

import numpy as np
import pandas as pd

from src import config


def load_raw(path=config.RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.generate_data` first.")
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[config.PRODUCT_COL].isna().any():
        raise ValueError("product_id contains missing values")
    if pd.to_datetime(df[config.DATE_COL], errors="coerce").isna().any():
        raise ValueError("date column contains unparseable dates")
    units = pd.to_numeric(df[config.TARGET], errors="coerce")
    if units.isna().all():
        raise ValueError("units_sold has no numeric values")
    if not set(df[config.PROMO_COL].dropna().unique()) <= {0, 1}:
        raise ValueError("promo must be 0 or 1")


def clean(df: pd.DataFrame):
    """Return (panel, report).  The report counts every fix, for the evidence trail."""
    validate_schema(df)
    df = df.copy()
    df.columns = df.columns.str.strip()
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    df[config.TARGET] = pd.to_numeric(df[config.TARGET], errors="coerce")
    report = {"raw_rows": int(len(df))}

    before = len(df)
    df = df.drop_duplicates(subset=[config.DATE_COL, config.PRODUCT_COL], keep="first")
    report["duplicates_removed"] = int(before - len(df))

    neg = df[config.TARGET] < 0
    report["negative_values_clipped_to_0"] = int(neg.sum())
    df.loc[neg, config.TARGET] = 0

    names = df.groupby(config.PRODUCT_COL)[config.NAME_COL].first()
    full_index = pd.date_range(df[config.DATE_COL].min(), df[config.DATE_COL].max(), freq="D")
    frames, missing_total = [], 0
    for pid, g in df.groupby(config.PRODUCT_COL):
        g = g.set_index(config.DATE_COL).sort_index().reindex(full_index)
        was_missing = g[config.TARGET].isna()
        missing_total += int(was_missing.sum())
        units = g[config.TARGET].interpolate(method="linear", limit_direction="both").round()
        out = pd.DataFrame({
            config.DATE_COL: full_index, config.PRODUCT_COL: pid,
            config.NAME_COL: names[pid], "units": units.values.astype(int),
            config.PROMO_COL: g[config.PROMO_COL].fillna(0).astype(int).values,
            "was_missing": was_missing.values.astype(int),
        })
        frames.append(out)
    panel = pd.concat(frames, ignore_index=True)
    report["missing_days_filled_by_interpolation"] = missing_total
    report["products"] = int(panel[config.PRODUCT_COL].nunique())
    report["days_per_product"] = int(len(full_index))
    return panel, report


def make_split_info(panel: pd.DataFrame) -> dict:
    last = panel[config.DATE_COL].max()
    test_start = last - pd.Timedelta(days=config.TEST_DAYS - 1)
    val_end = test_start - pd.Timedelta(days=1)
    val_start = val_end - pd.Timedelta(days=config.VAL_DAYS - 1)
    train_end = val_start - pd.Timedelta(days=1)
    fmt = lambda d: d.strftime("%Y-%m-%d")           # noqa: E731
    return {"train_start": fmt(panel[config.DATE_COL].min()), "train_end": fmt(train_end),
            "val_start": fmt(val_start), "val_end": fmt(val_end),
            "test_start": fmt(test_start), "test_end": fmt(last)}


def load_panel() -> pd.DataFrame:
    if not config.PANEL_PATH.exists():
        raise FileNotFoundError(f"{config.PANEL_PATH} missing - run `python -m src.data_prep` first.")
    return pd.read_csv(config.PANEL_PATH, parse_dates=[config.DATE_COL])


def load_split_info() -> dict:
    with open(config.SPLIT_PATH) as f:
        return json.load(f)


def product_series(panel: pd.DataFrame, pid: str):
    g = panel[panel[config.PRODUCT_COL] == pid].set_index(config.DATE_COL).sort_index()
    return g["units"], g[config.PROMO_COL]


def main() -> None:
    raw = load_raw()
    print(f"Raw data: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    panel, report = clean(raw)
    for k, v in report.items():
        print(f"  {k}: {v}")
    split = make_split_info(panel)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(config.PANEL_PATH, index=False)
    with open(config.SPLIT_PATH, "w") as f:
        json.dump(split, f, indent=2)
    with open(config.REPORTS_DIR / "cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Time-ordered split:")
    for name in ("train", "val", "test"):
        a, b = split.get(f"{name}_start"), split.get(f"{name}_end")
        n = (pd.Timestamp(b) - pd.Timestamp(a)).days + 1
        print(f"  {name:5s}: {a} to {b}  ({n} days)")

    # demo inputs for predict.py: current stock and a planned promotion
    last = panel[config.DATE_COL].max()
    recent = panel[panel[config.DATE_COL] > last - pd.Timedelta(days=28)]
    avg = recent.groupby(config.PRODUCT_COL)["units"].mean()
    pd.DataFrame({"product_id": avg.index, "on_hand": (avg * 6).round().astype(int).values,
                  "on_order": 0}).to_csv(config.PROCESSED_DIR / "sample_inventory.csv", index=False)
    future = pd.date_range(last + pd.Timedelta(days=1), periods=config.HORIZON)
    promo = pd.DataFrame({"date": future, "product_id": "SKU-001",
                          "promo": [1 if 2 <= i <= 8 else 0 for i in range(config.HORIZON)]})
    promo.to_csv(config.PROCESSED_DIR / "sample_future_promos.csv", index=False)
    print(f"Saved panel, split info and demo inputs to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
