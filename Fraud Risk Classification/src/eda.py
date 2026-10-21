"""Exploratory data analysis on the TRAIN split only (never looks at test).

Run:  python -m src.eda
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.data_prep import load_split
from src.features import add_features


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train = load_split("train")
    df = add_features(train.drop(columns=[config.TARGET]))
    df[config.TARGET] = train[config.TARGET].values
    rate = df[config.TARGET].mean()
    miss = df.isna().sum()

    lines = ["# EDA summary (train split)", "",
             f"- Transactions: {len(df):,} | frauds: {int(df[config.TARGET].sum())} | fraud rate **{rate:.2%}**",
             f"- A model that never flags anything is {1 - rate:.1%} accurate -> accuracy is useless; use PR-AUC, precision, recall.",
             f"- Missing values: {miss[miss > 0].to_dict() or 'none'}", ""]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    cats = ["merchant_category", "channel", "is_international", "device_changed", "is_night", "is_new_merchant"]
    for ax, col in zip(axes.ravel(), cats):
        r = df.groupby(col)[config.TARGET].mean().sort_values() * 100
        r.plot.barh(ax=ax, color="#2b6cb0")
        ax.axvline(rate * 100, color="red", ls="--", lw=1)
        ax.set_title(f"Fraud rate (%) by {col}"); ax.set_ylabel("")
        lines.append(f"- Fraud rate by **{col}**: " + ", ".join(f"{k}={v:.1f}%" for k, v in r.items()))
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "eda_fraud_by_segment.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(axes, ["amount_ratio", "distance_from_home_km", "txn_count_24h"],
                              ["Amount vs customer's usual", "Distance from home (km)", "Transactions in last 24h"]):
        upper = df[col].quantile(0.99)
        bins = np.linspace(0, upper, 40)
        ax.hist(df.loc[df[config.TARGET] == 0, col].dropna().clip(upper=upper), bins=bins, density=True,
                alpha=0.6, color="#2b6cb0", label="Legit")
        ax.hist(df.loc[df[config.TARGET] == 1, col].dropna().clip(upper=upper), bins=bins, density=True,
                alpha=0.6, color="#c53030", label="Fraud")
        ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "eda_distributions.png", dpi=150); plt.close(fig)

    med = df.groupby(config.TARGET)[["amount", "amount_ratio", "distance_from_home_km", "txn_count_24h"]].median().round(2)
    lines += ["", "Median values (0 = legit, 1 = fraud):", "", "```", med.to_string(), "```", ""]
    (config.REPORTS_DIR / "eda_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
