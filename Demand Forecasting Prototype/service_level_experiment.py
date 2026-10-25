import json

import pandas as pd

from src import config
from src.data_prep import load_panel, load_split_info, product_series
from src.inventory import simulate_policy
from src.inventory_sim import forecasts_by_review

panel, split = load_panel(), load_split_info()
bt = pd.read_csv(config.REPORTS_DIR / "test_backtest.csv", parse_dates=["origin", "date"])
sel = json.load(open(config.SELECTED_MODEL_PATH))
model = sel["selected_model"]
sigma = sel["daily_error_rmse_by_model"][model]

print(f"Model: {model}")
print("target service | actual fill rate | avg units in stock | holding cost | lost-sales cost | total cost")
for level in (0.80, 0.90, 0.95, 0.99):
    lost = demand_total = stock = hold = stockout = 0.0
    for pid in sorted(bt.product_id.unique()):
        y, _ = product_series(panel, pid)
        demand = y.loc[split["test_start"]:split["test_end"]].values
        fcs = forecasts_by_review(bt[(bt.model == model) & (bt.product_id == pid)])
        r = simulate_policy(demand, fcs, sigma[pid], service_level=level)
        lost += r["lost_units"]; demand_total += r["total_demand"]
        stock += r["avg_on_hand"]; hold += r["holding_cost"]; stockout += r["stockout_cost"]
    print(f"{level:>14.0%} | {1 - lost / demand_total:>16.1%} | {stock:>18,.0f} | "
          f"{hold:>12,.0f} | {stockout:>15,.0f} | {hold + stockout:>10,.0f}")