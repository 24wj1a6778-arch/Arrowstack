"""Generate SIMULATED daily sales for 5 products (2022-01-01 .. 2025-06-30).

Why simulate?  The handbook requires public, simulated or authorized data.
Real retail sales are confidential; simulation lets us build in known patterns
(trend, weekly and yearly seasonality, promotions, holidays, a slow-moving
intermittent item) and realistic data problems, so the project is runnable
offline and every result can be explained.

Products
  SKU-001 Bottled Water     high volume, strong summer peak, weekend lift
  SKU-002 Instant Noodles   growing trend, winter peak
  SKU-003 School Notebook   weekday-heavy, back-to-school spikes
  SKU-004 Phone Charger     small volume, gift-season peak in Nov/Dec
  SKU-005 Specialty Spice   slow mover, many zero-sales days (intermittent)

Data problems injected on purpose (the cleaning step must handle them):
  * a few missing dates, a few duplicated rows, a few negative values (returns glitch)

Run:  python -m src.generate_data
"""
import argparse

import numpy as np
import pandas as pd

from src import config
from src.features import holiday_flags

START, END = "2022-01-01", "2025-06-30"

PRODUCTS = {
    "SKU-001": dict(name="Bottled Water", base=120, trend=0.05,
                    week=[0.95, 0.95, 0.95, 1.0, 1.10, 1.15, 1.10],
                    yearly=("sin", 0.30, 200), promo=0.35, holiday=0.10, k=30),
    "SKU-002": dict(name="Instant Noodles", base=60, trend=0.12,
                    week=[1.0, 1.0, 1.0, 1.0, 1.05, 1.10, 1.05],
                    yearly=("sin", 0.20, 20), promo=0.40, holiday=0.05, k=30),
    "SKU-003": dict(name="School Notebook", base=35, trend=0.0,
                    week=[1.15, 1.15, 1.10, 1.10, 1.05, 0.70, 0.60],
                    yearly=("bumps", [(185, 20, 0.60), (10, 12, 0.25)]),
                    promo=0.30, holiday=-0.30, k=25),
    "SKU-004": dict(name="Phone Charger", base=18, trend=0.03,
                    week=[1.0, 1.0, 1.0, 1.0, 1.05, 1.10, 1.0],
                    yearly=("bumps", [(335, 22, 0.50)]), promo=0.60, holiday=0.25, k=20),
    "SKU-005": dict(name="Specialty Spice", base=2.2, trend=0.0,
                    week=[1, 1, 1, 1, 1, 1, 1], yearly=("none",),
                    promo=0.50, holiday=0.0, k=None),
}


def _yearly(kind, doy, years):
    if kind[0] == "none":
        return np.ones_like(doy, dtype=float)
    if kind[0] == "sin":
        _, amp, peak = kind
        return 1 + amp * np.cos(2 * np.pi * (doy - peak) / 365.25)
    out = np.ones_like(doy, dtype=float)
    for centre, width, amp in kind[1]:
        d = np.minimum(np.abs(doy - centre), 365 - np.abs(doy - centre))
        out += amp * np.exp(-0.5 * (d / width) ** 2)
    return out


def _promo_flags(n, rng):
    flags = np.zeros(n, dtype=int)
    i = 0
    while i < n:
        if rng.random() < 1 / 55:
            flags[i:i + 7] = 1
            i += 14                       # gap before the next promo can start
        else:
            i += 1
    return flags


def generate(seed: int = config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="D")
    n = len(dates)
    t_years = np.arange(n) / 365.25
    hol = holiday_flags(dates).astype(float)
    frames = []
    for pid, p in PRODUCTS.items():
        promo = _promo_flags(n, rng)
        mu = (p["base"] * (1 + p["trend"] * t_years)
              * np.array(p["week"])[dates.dayofweek.values]
              * _yearly(p["yearly"], dates.dayofyear.values, dates.year.values)
              * (1 + p["promo"] * promo) * (1 + p["holiday"] * hol))
        lam = mu if p["k"] is None else mu * rng.gamma(p["k"], 1 / p["k"], n)
        units = rng.poisson(lam)
        frames.append(pd.DataFrame({
            "date": dates, "product_id": pid, "product_name": p["name"],
            "units_sold": units, "promo": promo}))
    df = pd.concat(frames, ignore_index=True)

    # ---- realistic data problems ------------------------------------------------
    drop = rng.random(len(df)) < 0.004                       # missing dates
    dup = df[~drop].sample(6, random_state=seed)             # duplicated rows
    df = pd.concat([df[~drop], dup], ignore_index=True)
    neg_idx = rng.choice(df.index[df["units_sold"] > 5], size=5, replace=False)
    df.loc[neg_idx, "units_sold"] = -df.loc[neg_idx, "units_sold"]   # returns glitch
    return df.sort_values(["product_id", "date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--force", action="store_true", help="overwrite existing file")
    args = parser.parse_args()
    if config.RAW_PATH.exists() and not args.force:
        print(f"{config.RAW_PATH} already exists - keeping it (use --force to overwrite).")
        return
    config.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.seed)
    df.to_csv(config.RAW_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {config.RAW_PATH}  "
          f"({df['product_id'].nunique()} products, {df['date'].min().date()} to {df['date'].max().date()})")


if __name__ == "__main__":
    main()
