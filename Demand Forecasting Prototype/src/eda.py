"""Exploratory analysis of the TRAINING period only (never looks at validation/test).

Run:  python -m src.eda
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data_prep import load_panel, load_split_info
from src.features import holiday_flags

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    panel, split = load_panel(), load_split_info()
    train = panel[panel[config.DATE_COL] <= split["train_end"]].copy()
    train["dow"] = train[config.DATE_COL].dt.dayofweek
    train["month"] = train[config.DATE_COL].dt.month
    train["holiday"] = holiday_flags(train[config.DATE_COL])
    pids = sorted(train[config.PRODUCT_COL].unique())
    names = train.groupby(config.PRODUCT_COL)[config.NAME_COL].first()

    lines = ["# EDA summary (training period)", "",
             f"- Period: {split['train_start']} to {split['train_end']} | products: {len(pids)}", ""]
    lines += ["| product | mean/day | weekend vs weekday | promo uplift | zero-sales days | peak month |", "|---|---|---|---|---|---|"]
    for pid in pids:
        g = train[train[config.PRODUCT_COL] == pid]
        weekend = g[g.dow >= 5]["units"].mean() / g[g.dow < 5]["units"].mean()
        promo = g[g.promo == 1]["units"].mean() / g[g.promo == 0]["units"].mean()
        zero = (g["units"] == 0).mean()
        peak = g.groupby("month")["units"].mean().idxmax()
        lines.append(f"| {pid} {names[pid]} | {g['units'].mean():.1f} | x{weekend:.2f} | x{promo:.2f} | {zero:.0%} | {peak} |")
    lines += ["", "Note: the 'promo uplift' compares average sales on promo and non-promo days; promos are not "
              "randomly timed, so treat it as indicative, not causal.", ""]

    fig, axes = plt.subplots(len(pids), 1, figsize=(11, 2.3 * len(pids)), sharex=True)
    for ax, pid in zip(axes, pids):
        g = train[train[config.PRODUCT_COL] == pid].set_index(config.DATE_COL)
        ax.plot(g.index, g["units"], color="#cbd5e0", lw=0.6)
        ax.plot(g.index, g["units"].rolling(28, center=True).mean(), color="#2b6cb0", lw=1.5)
        ax.set_ylabel(f"{pid}\n{names[pid]}", fontsize=8)
    axes[0].set_title("Daily units (grey) and 28-day average (blue)")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "eda_series.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for pid in pids:
        g = train[train[config.PRODUCT_COL] == pid]
        axes[0].plot(WEEKDAYS, g.groupby("dow")["units"].mean() / g["units"].mean(), marker="o", label=pid)
        axes[1].plot(range(1, 13), g.groupby("month")["units"].mean() / g["units"].mean(), marker="o", label=pid)
    axes[0].set_title("Weekly pattern (1.0 = average day)"); axes[1].set_title("Yearly pattern (1.0 = average month)")
    axes[1].set_xlabel("Month"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "eda_seasonality.png", dpi=150); plt.close(fig)

    # autocorrelation at a few lags for the biggest product
    y = train[train[config.PRODUCT_COL] == pids[0]]["units"].values.astype(float)
    acf = {lag: float(np.corrcoef(y[:-lag], y[lag:])[0, 1]) for lag in (1, 7, 14, 28, 364)}
    lines += ["Autocorrelation of SKU-001 at lags (days): " + ", ".join(f"{k}: {v:.2f}" for k, v in acf.items()),
              "(strong at 7, 14, 28 = weekly pattern; at 364 = yearly pattern)", ""]
    (config.REPORTS_DIR / "eda_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
