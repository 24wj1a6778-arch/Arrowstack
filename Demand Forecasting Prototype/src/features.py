"""Calendar features and the feature builder for the direct GBM forecaster.

Leakage rule: every feature for a (forecast origin o, horizon h) row is computed
from sales up to day o only.  Calendar facts and *planned* promotions for the
target day o+h are legitimately known in advance.
"""
import numpy as np
import pandas as pd

from src import config

FIXED_HOLIDAYS = {(1, 1), (2, 14), (5, 1), (8, 15), (10, 31), (12, 24), (12, 25), (12, 31)}

FEATURE_NAMES = [
    "h", "dow", "month", "doy_sin", "doy_cos", "is_holiday", "promo", "promo_last7",
    "mean7", "mean28", "mean91", "trend_ratio", "same_weekday_mean", "last_year_mean7",
]


def holiday_flags(dates) -> np.ndarray:
    """1 on fixed-date holidays and on 'Black Friday' (the Friday after the 4th Thursday of November)."""
    d = pd.DatetimeIndex(dates)
    fixed = np.array([(m, day) in FIXED_HOLIDAYS for m, day in zip(d.month, d.day)])
    black_friday = (d.dayofweek == 4) & (d.month == 11) & (d.day >= 23) & (d.day <= 29)
    return (fixed | black_friday).astype(int)


def direct_feature_matrix(y, promo, dates, origins, horizons) -> pd.DataFrame:
    """One row per (origin, horizon) pair.

    y       sales history (only indices <= origin are ever read)
    promo   promo flag for every date in `dates` (history + planned future)
    dates   DatetimeIndex covering history and the forecast horizon
    """
    y = np.asarray(y, dtype=float)
    promo = np.asarray(promo, dtype=float)
    o = np.asarray(origins, dtype=int)
    h = np.asarray(horizons, dtype=int)
    t = o + h
    dates = pd.DatetimeIndex(dates)

    cs = np.concatenate([[0.0], np.cumsum(y)])                    # cs[i] = sum(y[:i])
    pcs = np.concatenate([[0.0], np.cumsum(promo[: len(y)])])

    def wmean(cum, w):
        start = np.maximum(o + 1 - w, 0)
        return (cum[o + 1] - cum[start]) / (o + 1 - start)

    m7, m28, m91 = wmean(cs, 7), wmean(cs, 28), wmean(cs, 91)

    # same weekday as the target day, using only lags that are already observed
    kmin = np.ceil(h / 7).astype(int)
    acc, cnt = np.zeros(len(t)), np.zeros(len(t))
    for j in range(4):
        idx = t - 7 * (kmin + j)
        ok = idx >= 0
        acc += np.where(ok, y[np.clip(idx, 0, None)], 0.0)
        cnt += ok
    same_wd = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)

    # same time last year (7-day mean around t-364), only if fully observed
    idx = t - 364
    ok = (idx - 3 >= 0) & (idx + 3 <= o)
    lo, hi = np.clip(idx - 3, 0, None), np.clip(idx + 4, 0, len(y))
    last_year = np.where(ok, (cs[hi] - cs[lo]) / 7.0, np.nan)

    td = dates[t]
    doy = td.dayofyear.values
    hol = holiday_flags(dates)[t]
    return pd.DataFrame({
        "h": h,
        "dow": td.dayofweek.values,
        "month": td.month.values,
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "is_holiday": hol,
        "promo": promo[t],
        "promo_last7": wmean(pcs, 7),
        "mean7": m7, "mean28": m28, "mean91": m91,
        "trend_ratio": m28 / (m91 + 1e-6),
        "same_weekday_mean": same_wd,
        "last_year_mean7": last_year,
    })[FEATURE_NAMES]
