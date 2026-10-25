import pandas as pd

bt = pd.read_csv("reports/test_backtest.csv")
bt["ae"] = (bt.actual - bt.forecast).abs()
windows = bt.groupby(["model", "product_id", "origin"])["ae"].mean().reset_index()
table = windows.pivot(index=["product_id", "origin"], columns="model", values="ae")

for other in ["holt_winters", "seasonal_mean", "moving_average_28", "seasonal_naive"]:
    wins = int((table["gbm_direct"] < table[other]).sum())
    print(f"gbm_direct beats {other} in {wins} of {len(table)} forecast windows")