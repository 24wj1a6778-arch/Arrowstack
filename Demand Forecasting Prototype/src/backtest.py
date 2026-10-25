"""Rolling-origin (walk-forward) backtest.

Every week we pretend to stand on a "forecast origin" day, show each model ONLY the
history up to that day, and ask for the next 14 days.  Then we compare with what
really happened.  This mimics how the forecast would be used in practice.

Leakage guard: the history passed to a model is sliced with `.loc[:origin]`.
Only planned promotions for the horizon are passed as extra inputs.
"""
import pandas as pd

from src import config
from src.data_prep import product_series


def origins_for(start, end, stride=config.STRIDE, horizon=config.HORIZON):
    """Origins (last observed day) whose full forecast window fits inside [start, end].

    The first window starts on `start`; a new origin is added every `stride` days.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out, origin = [], start - pd.Timedelta(days=1)
    while origin + pd.Timedelta(days=horizon) <= end:
        out.append(origin)
        origin += pd.Timedelta(days=stride)
    return out


def rolling_origin_backtest(panel, factories: dict, origins, horizon=config.HORIZON,
                            products=None) -> pd.DataFrame:
    rows = []
    pids = products or sorted(panel[config.PRODUCT_COL].unique())
    for pid in pids:
        y_all, promo_all = product_series(panel, pid)
        for origin in origins:
            hist, hist_promo = y_all.loc[:origin], promo_all.loc[:origin]
            fut_idx = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon)
            fut_promo = promo_all.reindex(fut_idx).fillna(0).values      # planned, known
            actual = y_all.reindex(fut_idx).values
            for name, factory in factories.items():
                model = factory().fit(hist, hist_promo)
                f = model.predict(horizon, fut_promo)
                rows.append(pd.DataFrame({
                    "model": name, "product_id": pid, "origin": origin,
                    "h": range(1, horizon + 1), "date": fut_idx,
                    "actual": actual, "forecast": f, "promo": fut_promo}))
    return pd.concat(rows, ignore_index=True)
