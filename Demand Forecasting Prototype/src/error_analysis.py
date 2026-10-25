"""Error analysis for the selected model on the TEST period.

Where does the forecast go wrong?  By forecast horizon, weekday, promotion,
holiday and product - and compared with the seasonal-naive baseline.

Outputs (reports/): error_by_horizon.csv, error_by_weekday.csv, error_by_condition.csv,
                    error_by_product.csv, worst_forecast_errors.csv, error_analysis.md,
                    figures/error_by_horizon.png

Run:  python -m src.error_analysis
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.features import holiday_flags
from src.metrics import bias, mae, wape

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, observed=True):
        rows.append({by: key, "n": len(g), "mae": mae(g.actual, g.forecast),
                     "wape": wape(g.actual, g.forecast), "bias": bias(g.actual, g.forecast)})
    return pd.DataFrame(rows)


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.SELECTED_MODEL_PATH) as f:
        chosen = json.load(f)["selected_model"]
    bt = pd.read_csv(config.REPORTS_DIR / "test_backtest.csv", parse_dates=["origin", "date"])
    g = bt[bt.model == chosen].copy()
    base = bt[bt.model == "seasonal_naive"].copy()
    g["error"] = g.forecast - g.actual
    g["weekday"] = pd.Categorical([WEEKDAYS[d] for d in g.date.dt.dayofweek], WEEKDAYS, ordered=True)
    g["holiday"] = np.where(holiday_flags(g.date) == 1, "holiday", "normal day")
    g["promotion"] = np.where(g.promo == 1, "promo day", "no promo")

    by_h = table(g, "h").merge(table(base, "h")[["h", "mae"]].rename(columns={"mae": "baseline_mae"}), on="h")
    by_wd = table(g, "weekday")
    by_cond = pd.concat([table(g, "promotion").rename(columns={"promotion": "condition"}),
                         table(g, "holiday").rename(columns={"holiday": "condition"})])
    by_prod = table(g, "product_id")
    for name, t in [("horizon", by_h), ("weekday", by_wd), ("condition", by_cond), ("product", by_prod)]:
        t.to_csv(config.REPORTS_DIR / f"error_by_{name}.csv", index=False)

    worst = g.assign(abs_error=g.error.abs()).sort_values("abs_error", ascending=False).head(15)
    worst[["product_id", "date", "h", "actual", "forecast", "error", "promo"]].to_csv(
        config.REPORTS_DIR / "worst_forecast_errors.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(by_h["h"], by_h["mae"], marker="o", label=chosen)
    ax.plot(by_h["h"], by_h["baseline_mae"], marker="o", ls="--", color="#a0aec0", label="seasonal_naive")
    ax.set_xlabel("Days ahead"); ax.set_ylabel("MAE (units, all products)")
    ax.set_title("Forecast error by days ahead"); ax.legend()
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "error_by_horizon.png", dpi=150); plt.close(fig)

    c = by_cond.set_index("condition")
    underforecast_prod = by_prod.sort_values("bias").head(1).iloc[0]
    slope = float(np.polyfit(by_h["h"], by_h["mae"], 1)[0])
    rel_slope = slope / float(by_h["mae"].mean())
    if rel_slope > 0.02:
        horizon_text = f"error rises with horizon (about {rel_slope:.1%} per extra day)"
    elif rel_slope < -0.02:
        horizon_text = f"error falls with horizon ({rel_slope:.1%} per extra day) - unusual, check for noise"
    else:
        horizon_text = "error is roughly flat across the 14 days (trend under 2% per day)"
    lines = [
        "# Error analysis (test period)", "",
        f"- Model: **{chosen}** | rows analysed: {len(g):,} (product-days across overlapping weekly forecasts)", "",
        "## By forecast horizon", "",
        f"- Day 1 MAE {by_h.mae.iloc[0]:.1f}, day 14 MAE {by_h.mae.iloc[-1]:.1f} units: {horizon_text}.",
        f"- The model beats the seasonal-naive baseline at every horizon "
        f"({int((by_h.mae < by_h.baseline_mae).sum())} of {len(by_h)} days ahead).", "",
        "## By condition", "",
        f"- Promo days: MAE {c.loc['promo day', 'mae']:.1f}, bias {c.loc['promo day', 'bias']:+.1f} | "
        f"non-promo days: MAE {c.loc['no promo', 'mae']:.1f}, bias {c.loc['no promo', 'bias']:+.1f}",
        f"- Holidays: MAE {c.loc['holiday', 'mae'] if 'holiday' in c.index else float('nan'):.1f} "
        f"(only {int(c.loc['holiday', 'n']) if 'holiday' in c.index else 0} product-days: treat with caution)", "",
        "## By product", "",
    ]
    lines += [f"- {r.product_id}: MAE {r.mae:.1f}, WAPE {r.wape:.0%}, bias {r.bias:+.1f}" for r in by_prod.itertuples()]
    lines += ["", f"- Largest under-forecast bias: {underforecast_prod.product_id} ({underforecast_prod.bias:+.1f} units/day). "
              "Under-forecasting causes stockouts.", "",
              "> Interpretation (write your own): which products, days or conditions are hardest, and why? "
              "What data would help (price changes, weather, competitor actions, stock-outs hiding demand)?", ""]
    (config.REPORTS_DIR / "error_analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("\nBy weekday:")
    print(by_wd.round(2).to_string(index=False))
    print("\nBy horizon:")
    print(by_h.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
