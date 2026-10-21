"""Final evaluation on the untouched TEST split, with bootstrap uncertainty.

Fraud is rare, so the test set holds only ~140 frauds and a single number
hides a lot of noise.  We report a 95% bootstrap interval for PR-AUC.

Run:  python -m src.evaluate
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             precision_recall_curve)

from src import config
from src.data_prep import load_split, xy
from src.metrics import classification_metrics, total_cost
from src.train import dummy_model


def bootstrap_pr_auc(y, proba, n=300, seed=config.SEED):
    rng = np.random.default_rng(seed)
    y, proba = np.asarray(y), np.asarray(proba)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        scores.append(average_precision_score(y[idx], proba[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(config.FINAL_MODEL_PATH)
    with open(config.THRESHOLD_PATH) as f:
        info = json.load(f)
    thr = info["threshold"]

    X_train, y_train = xy(load_split("train"))
    X_test, y_test = xy(load_split("test"))
    amount = X_test[config.AMOUNT_COL].values
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= thr).astype(int)

    dummy = dummy_model().fit(X_train, y_train)
    d_proba = dummy.predict_proba(X_test)[:, 1]

    sel = classification_metrics(y_test, proba, thr, amount)
    lo, hi = bootstrap_pr_auc(y_test, proba)
    results = {
        "algorithm": info["algorithm"], "strategy": info["strategy"],
        "n_test": int(len(y_test)), "n_test_frauds": int(y_test.sum()),
        "test_fraud_rate": float(y_test.mean()),
        "selected": sel,
        "pr_auc_95ci": [lo, hi],
        "dummy_baseline": classification_metrics(y_test, d_proba, 0.5, amount),
        "strategies_cost": {
            "approve_everything": total_cost(y_test, np.zeros(len(y_test), dtype=int), amount),
            "review_everything": total_cost(y_test, np.ones(len(y_test), dtype=int), amount),
            "use_model": sel["total_cost"],
        },
        "fraud_amount_total": float(amount[y_test.values == 1].sum()),
        "fraud_amount_caught": float(amount[(y_test.values == 1) & (pred == 1)].sum()),
    }
    with open(config.REPORTS_DIR / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Test results: {info['algorithm']} + {info['strategy']}  (threshold {thr:.4f})")
    print(f"  Test set: {len(y_test):,} transactions, {int(y_test.sum())} frauds")
    print(f"  PR-AUC {sel['pr_auc']:.3f}  (95% CI {lo:.3f}-{hi:.3f}; baseline {results['dummy_baseline']['pr_auc']:.3f}) | ROC-AUC {sel['roc_auc']:.3f}")
    print(f"  Precision {sel['precision']:.3f} | Recall {sel['recall']:.3f} | Alerts per 1,000: {sel['alerts_per_1000']:.1f}")
    print(f"  Confusion: TP={sel['tp']} FP={sel['fp']} FN={sel['fn']} TN={sel['tn']}")
    print(f"  Total cost: {results['strategies_cost']}")
    print(f"  Fraud $ caught: {results['fraud_amount_caught']:,.0f} of {results['fraud_amount_total']:,.0f}")

    # ---- figures ---------------------------------------------------------------
    prec, rec, _ = precision_recall_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(rec, prec, label=f"model (AP = {sel['pr_auc']:.2f})")
    ax.axhline(y_test.mean(), color="k", ls="--", lw=1, label="baseline (fraud rate)")
    ax.scatter([sel["recall"]], [sel["precision"]], color="red", zorder=5, label="chosen threshold")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall curve (test)"); ax.legend()
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "pr_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Legit", "Fraud"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (test)")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)

    c = results["strategies_cost"]
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["Approve everything", "Review everything", "Use the model"]
    vals = [c["approve_everything"], c["review_everything"], c["use_model"]]
    ax.bar(names, vals, color=["#a0aec0", "#a0aec0", "#2b6cb0"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom")
    ax.set_ylabel("Total cost on test set"); ax.set_title("Cost of each strategy (lower is better)")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "cost_comparison.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(proba[y_test.values == 0], bins=bins, alpha=0.6, label="Legit", color="#2b6cb0")
    ax.hist(proba[y_test.values == 1], bins=bins, alpha=0.8, label="Fraud", color="#c53030")
    ax.axvline(thr, color="k", ls="--", label="threshold"); ax.set_yscale("log")
    ax.set_xlabel("Predicted fraud score"); ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Score distribution (test)"); ax.legend()
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "score_distribution.png", dpi=150); plt.close(fig)
    print(f"Figures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
