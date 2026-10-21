"""Threshold analysis on the VALIDATION set: the precision/recall trade-off.

A fraud team does not think in probabilities; it thinks in "how many alerts can
we review per day?".  This script turns the model into operating points:

  alert budget -> threshold -> precision, recall, alerts per 1,000, cost

and marks the cost-optimal point chosen in train.py.

Outputs: reports/operating_points_val.csv, reports/threshold_analysis.md,
         reports/figures/threshold_tradeoff.png

Run:  python -m src.threshold_analysis
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data_prep import load_split, xy
from src.metrics import (classification_metrics, total_cost, threshold_for_alert_budget)


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(config.FINAL_MODEL_PATH)
    with open(config.THRESHOLD_PATH) as f:
        info = json.load(f)
    X_val, y_val = xy(load_split("val"))
    amount = X_val[config.AMOUNT_COL].values
    proba = model.predict_proba(X_val)[:, 1]

    rows = []
    for b in config.ALERT_BUDGETS:
        thr = threshold_for_alert_budget(proba, b)
        m = classification_metrics(y_val, proba, thr, amount)
        rows.append({"operating_point": f"alert budget {b:.2%}", **m})
    chosen = classification_metrics(y_val, proba, info["threshold"], amount)
    rows.append({"operating_point": "COST-OPTIMAL (chosen)", **chosen})
    table = pd.DataFrame(rows)
    keep = ["operating_point", "threshold", "alerts_per_1000", "precision", "recall", "tp", "fp", "fn", "total_cost"]
    table[keep].to_csv(config.REPORTS_DIR / "operating_points_val.csv", index=False)
    print(table[keep].round(3).to_string(index=False))

    # curves over alert rate
    fracs = np.linspace(0.001, 0.08, 120)
    prec, rec, cost = [], [], []
    for fr in fracs:
        t = threshold_for_alert_budget(proba, fr)
        pred = (proba >= t).astype(int)
        tp = ((pred == 1) & (y_val.values == 1)).sum()
        prec.append(tp / max(pred.sum(), 1))
        rec.append(tp / y_val.sum())
        cost.append(total_cost(y_val, pred, amount))
    chosen_rate = chosen["alerts_per_1000"] / 1000

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fracs * 100, prec, label="Precision")
    axes[0].plot(fracs * 100, rec, label="Recall")
    axes[0].axvline(chosen_rate * 100, color="red", ls="--", label="chosen")
    axes[0].set_xlabel("Share of transactions alerted (%)"); axes[0].set_ylabel("Score")
    axes[0].set_title("Precision vs recall as alerts increase"); axes[0].legend()
    axes[1].plot(fracs * 100, cost)
    axes[1].axvline(chosen_rate * 100, color="red", ls="--", label=f"chosen = {chosen_rate:.2%}")
    axes[1].set_xlabel("Share of transactions alerted (%)"); axes[1].set_ylabel("Total cost (validation)")
    axes[1].set_title("Total cost (missed fraud + review cost)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "threshold_tradeoff.png", dpi=150); plt.close(fig)

    lines = [
        "# Threshold analysis (validation set)", "",
        f"- Model: {info['algorithm']} ({info['strategy']})",
        f"- Cost-optimal threshold: **{info['threshold']:.4f}** -> "
        f"{chosen['alerts_per_1000']:.1f} alerts per 1,000 transactions, "
        f"precision {chosen['precision']:.2f}, recall {chosen['recall']:.2f}", "",
        "How to read it: the more transactions the team reviews, the more fraud is caught "
        "(recall up) but the more reviews are wasted (precision down). The cost-optimal point is "
        "where one more review stops paying for itself.", "",
        "See `operating_points_val.csv` for every operating point and "
        "`figures/threshold_tradeoff.png` for the curves.", "",
        "> Write your own recommendation: which operating point would you give the fraud manager "
        "if analysts could review at most 1% of transactions?", "",
    ]
    (config.REPORTS_DIR / "threshold_analysis.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
