"""Forecast the next 14 days and recommend order quantities.

Run:
  python -m src.predict --inventory-file data/processed/sample_inventory.csv \
                        --promo-file data/processed/sample_future_promos.csv

Outputs: reports/forecast_next14.csv and, if an inventory file is given,
         reports/order_recommendations.csv

The selected model is refitted on ALL history (cheap), then forecasts the 14 days
after the last date in the data.

Edge cases handled
  * short history (< 120 days)   -> falls back to a simpler model and says so
  * planned promotions missing   -> assumes no promotion
  * product not in the history   -> flagged "unknown_product", no order made
  * negative stock / on order    -> flagged "invalid_input", no order made
  * missing columns in a file    -> clear error naming them
"""
import argparse
import json

import numpy as np
import pandas as pd

from src import config
from src.data_prep import load_panel, product_series
from src.inventory import order_quantity, order_up_to_level, z_value
from src.models import MODEL_FACTORIES


def choose_model_for(history_days: int, selected: str):
    """Return (model_name, note).  Falls back when there is too little history."""
    needed = {"gbm_direct": config.MIN_HISTORY_DAYS, "holt_winters": 4 * config.SEASON}
    if history_days >= max(config.MIN_HISTORY_DAYS, needed.get(selected, 0)):
        return selected, ""
    if history_days >= 28:
        return "seasonal_mean", f"short history ({history_days} days): used seasonal_mean instead of {selected}"
    if history_days >= 1:
        return "moving_average_28", f"very short history ({history_days} days): used moving_average_28"
    raise ValueError("no history available")


def read_promo_file(path, pid, future_dates) -> np.ndarray:
    if path is None:
        return np.zeros(len(future_dates))
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [c for c in ("date", "product_id", "promo") if c not in df.columns]
    if missing:
        raise ValueError(f"Promo file is missing required columns: {missing}")
    if not set(df["promo"].dropna().unique()) <= {0, 1}:
        raise ValueError("promo values must be 0 or 1")
    s = df[df.product_id == pid].set_index("date")["promo"]
    return s.reindex(future_dates).fillna(0).values


def forecast_next(panel: pd.DataFrame, selected: str, sigma: dict, promo_file=None) -> pd.DataFrame:
    z90 = z_value(0.5 + config.INTERVAL_LEVEL / 2)
    out = []
    for pid in sorted(panel[config.PRODUCT_COL].unique()):
        y, promo = product_series(panel, pid)
        name, note = choose_model_for(len(y), selected)
        future = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=config.HORIZON)
        fut_promo = read_promo_file(promo_file, pid, future)
        f = MODEL_FACTORIES[name]().fit(y, promo).predict(config.HORIZON, fut_promo)
        s = sigma.get(name, {}).get(pid, float(y.iloc[-28:].std()))
        out.append(pd.DataFrame({
            "product_id": pid, "date": future, "h": range(1, config.HORIZON + 1),
            "forecast": np.round(f, 1), "lower_90": np.round(np.clip(f - z90 * s, 0, None), 1),
            "upper_90": np.round(f + z90 * s, 1), "planned_promo": fut_promo.astype(int),
            "model_used": name, "note": note, "daily_sigma": round(s, 2)}))
    return pd.concat(out, ignore_index=True)


def recommend_orders(fc: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ("product_id", "on_hand", "on_order") if c not in inventory.columns]
    if missing:
        raise ValueError(f"Inventory file is missing required columns: {missing}")
    rows = []
    known = set(fc["product_id"])
    for r in inventory.itertuples():
        base = {"product_id": r.product_id, "on_hand": r.on_hand, "on_order": r.on_order}
        stock = pd.to_numeric(pd.Series([r.on_hand, r.on_order]), errors="coerce")
        if r.product_id not in known:
            rows.append({**base, "status": "unknown_product", "order_quantity": np.nan})
        elif stock.isna().any() or (stock < 0).any():
            rows.append({**base, "status": "invalid_input", "order_quantity": np.nan})
        else:
            g = fc[fc.product_id == r.product_id].sort_values("h")
            sigma = float(g["daily_sigma"].iloc[0])
            level = order_up_to_level(g["forecast"].values, sigma)
            rows.append({**base, "forecast_14d": round(float(g["forecast"].sum()), 1),
                         "order_up_to_level": round(level, 1),
                         "order_quantity": order_quantity(level, float(stock[0]), float(stock[1])),
                         "status": "ok"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast the next 14 days and recommend orders")
    parser.add_argument("--promo-file", default=None, help="CSV: date, product_id, promo (planned promotions)")
    parser.add_argument("--inventory-file", default=None, help="CSV: product_id, on_hand, on_order")
    args = parser.parse_args()

    with open(config.SELECTED_MODEL_PATH) as f:
        sel = json.load(f)
    panel = load_panel()
    fc = forecast_next(panel, sel["selected_model"], sel["daily_error_rmse_by_model"], args.promo_file)
    fc.to_csv(config.REPORTS_DIR / "forecast_next14.csv", index=False)
    summary = fc.groupby("product_id").agg(forecast_14d=("forecast", "sum"), model=("model_used", "first"))
    print(f"Forecast for {fc['date'].min().date()} to {fc['date'].max().date()} (model: {sel['selected_model']})")
    print(summary.round(1).to_string())
    if fc["note"].ne("").any():
        print("Notes:", "; ".join(sorted(set(fc["note"]) - {""})))

    if args.inventory_file:
        orders = recommend_orders(fc, pd.read_csv(args.inventory_file))
        orders.to_csv(config.REPORTS_DIR / "order_recommendations.csv", index=False)
        print("\nOrder recommendations (order-up-to policy, "
              f"{config.SERVICE_LEVEL:.0%} service target):")
        print(orders.to_string(index=False))


if __name__ == "__main__":
    main()
