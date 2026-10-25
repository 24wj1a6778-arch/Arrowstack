"""Forecast accuracy metrics.

MAE   average absolute error, in units                       (easy to explain)
RMSE  penalises large misses more                            (what safety stock is based on)
WAPE  total absolute error / total demand                    (a % that works with zeros)
Bias  average (forecast - actual); >0 over-forecast          (drives overstock or stockouts)
MASE  MAE divided by the MAE of the in-sample seasonal naive (scale-free; <1 beats the baseline)

MAPE is deliberately NOT used: it explodes on days with zero sales.
"""
import numpy as np
import pandas as pd

from src import config


def mae(a, f):
    return float(np.mean(np.abs(np.asarray(f) - np.asarray(a))))


def rmse(a, f):
    return float(np.sqrt(np.mean((np.asarray(f) - np.asarray(a)) ** 2)))


def wape(a, f):
    a = np.asarray(a, dtype=float)
    total = a.sum()
    return float(np.abs(np.asarray(f) - a).sum() / total) if total > 0 else float("nan")


def bias(a, f):
    return float(np.mean(np.asarray(f) - np.asarray(a)))


def seasonal_naive_scale(y, m: int = config.SEASON) -> float:
    """In-sample MAE of the seasonal naive forecast (the MASE denominator)."""
    y = np.asarray(y, dtype=float)
    if len(y) <= m:
        return 1.0
    s = float(np.mean(np.abs(y[m:] - y[:-m])))
    return s if s > 0 else 1.0


def mase(a, f, scale: float) -> float:
    return mae(a, f) / scale


def summarize(bt: pd.DataFrame, scales: dict):
    """Metrics per (model, product) and per model from a backtest frame.

    `bt` has columns model, product_id, actual, forecast.  Rows with no actual are ignored.
    """
    bt = bt.dropna(subset=["actual"])
    rows = []
    for (model, pid), g in bt.groupby(["model", "product_id"]):
        rows.append({"model": model, "product_id": pid, "n": len(g),
                     "mae": mae(g.actual, g.forecast), "rmse": rmse(g.actual, g.forecast),
                     "wape": wape(g.actual, g.forecast), "bias": bias(g.actual, g.forecast),
                     "mase": mase(g.actual, g.forecast, scales[pid])})
    per_product = pd.DataFrame(rows)
    overall = []
    for model, g in bt.groupby("model"):
        pp = per_product[per_product.model == model]
        overall.append({"model": model, "mae": mae(g.actual, g.forecast),
                        "rmse": rmse(g.actual, g.forecast), "wape": wape(g.actual, g.forecast),
                        "bias": bias(g.actual, g.forecast), "mean_mase": float(pp["mase"].mean())})
    overall = pd.DataFrame(overall).sort_values("mean_mase").reset_index(drop=True)
    return overall, per_product
