"""Business impact: simulate the inventory policy on the TEST period for every model.

Same policy, same 95% service target, same costs - only the forecast (and its
error-based safety stock) changes.  This shows what better forecasting is worth.

Outputs: reports/inventory_results.csv, reports/inventory_by_product.csv,
         reports/inventory_summary.md, reports/figures/inventory_comparison.png

Run:  python -m src.inventory_sim
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data_prep import load_panel, load_split_info, product_series
from src.inventory import order_up_to_level, safety_stock, simulate_policy


def forecasts_by_review(bt_mp: pd.DataFrame):
    """List of 14-day forecast arrays, ordered by origin (= weekly review)."""
    return [g.sort_values("h")["forecast"].values for _, g in bt_mp.groupby("origin")]


def main() -> None:
    panel, split = load_panel(), load_split_info()
    bt = pd.read_csv(config.REPORTS_DIR / "test_backtest.csv", parse_dates=["origin", "date"])
    with open(config.SELECTED_MODEL_PATH) as f:
        sel = json.load(f)
    sigma = sel["daily_error_rmse_by_model"]
    chosen = sel["selected_model"]

    rows = []
    for model in bt["model"].unique():
        for pid in sorted(bt["product_id"].unique()):
            y, _ = product_series(panel, pid)
            demand = y.loc[split["test_start"]:split["test_end"]].values
            fcs = forecasts_by_review(bt[(bt.model == model) & (bt.product_id == pid)])
            res = simulate_policy(demand, fcs, sigma[model][pid])
            res.update(model=model, product_id=pid,
                       avg_safety_stock=safety_stock(sigma[model][pid]))
            rows.append(res)
    by_product = pd.DataFrame(rows)
    by_product.to_csv(config.REPORTS_DIR / "inventory_by_product.csv", index=False)

    agg = by_product.groupby("model").agg(
        total_demand=("total_demand", "sum"), lost_units=("lost_units", "sum"),
        avg_on_hand=("avg_on_hand", "sum"), holding_cost=("holding_cost", "sum"),
        stockout_cost=("stockout_cost", "sum"), total_cost=("total_cost", "sum")).reset_index()
    agg["fill_rate"] = 1 - agg["lost_units"] / agg["total_demand"]
    agg = agg.sort_values("total_cost").reset_index(drop=True)
    agg.to_csv(config.REPORTS_DIR / "inventory_results.csv", index=False)

    show = agg[["model", "fill_rate", "lost_units", "avg_on_hand", "holding_cost", "stockout_cost", "total_cost"]]
    print(f"Inventory simulation on test ({split['test_start']} to {split['test_end']}), "
          f"target service {config.SERVICE_LEVEL:.0%}, lead time {config.LEAD_TIME}d, review {config.REVIEW_PERIOD}d")
    print(show.round(3).to_string(index=False))

    base = agg.set_index("model").loc["moving_average_28"]
    best = agg.set_index("model").loc[chosen]
    saving = 1 - best["total_cost"] / base["total_cost"]
    print(f"\nSelected model ({chosen}) vs a 28-day moving-average planner: "
          f"cost {best['total_cost']:,.0f} vs {base['total_cost']:,.0f} ({abs(saving):.1%} {'lower' if saving > 0 else 'higher'}), "
          f"fill rate {best['fill_rate']:.1%} vs {base['fill_rate']:.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#2b6cb0" if m == chosen else "#a0aec0" for m in agg["model"]]
    axes[0].barh(agg["model"][::-1], agg["total_cost"][::-1], color=colors[::-1])
    axes[0].set_xlabel("Total cost (holding + lost sales)"); axes[0].set_title("Cost by forecasting model (lower is better)")
    axes[1].scatter(agg["avg_on_hand"], agg["fill_rate"] * 100, color=colors, s=70)
    for r in agg.itertuples():
        axes[1].annotate(r.model, (r.avg_on_hand, r.fill_rate * 100), fontsize=7, xytext=(4, 3), textcoords="offset points")
    axes[1].axhline(config.SERVICE_LEVEL * 100, color="red", ls="--", lw=1, label="target service")
    axes[1].set_xlabel("Average units in stock (all products)"); axes[1].set_ylabel("Fill rate (%)")
    axes[1].set_title("Service vs inventory"); axes[1].legend()
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "inventory_comparison.png", dpi=150); plt.close(fig)

    lines = ["# Inventory simulation (test period)", "",
             f"- Policy: order-up-to, review every {config.REVIEW_PERIOD} days, lead time {config.LEAD_TIME} days, "
             f"target service {config.SERVICE_LEVEL:.0%}",
             f"- Costs (assumed): holding {config.HOLDING_COST}/unit/day, lost sale {config.STOCKOUT_COST}/unit", "",
             "| model | fill rate | lost units | avg stock | total cost |", "|---|---|---|---|---|"]
    lines += [f"| {r.model} | {r.fill_rate:.1%} | {r.lost_units:,.0f} | {r.avg_on_hand:,.0f} | {r.total_cost:,.0f} |"
              for r in agg.itertuples()]
    lines += ["", "> Write your own reading: which model achieves the target service with the least stock? "
              "Which models miss the target, and why (hint: forecast bias)?", ""]
    (config.REPORTS_DIR / "inventory_summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
