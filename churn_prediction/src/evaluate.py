"""Final evaluation on the untouched TEST split + diagnostic plots.

Run:  python -m src.evaluate
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from src import config
from src.data_prep import load_split, xy
from src.metrics import business_value, classification_metrics, top_decile_lift


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(config.FINAL_MODEL_PATH)
    dummy = joblib.load(config.CANDIDATES_DIR / "dummy_baseline.joblib")
    with open(config.THRESHOLD_PATH) as f:
        info = json.load(f)
    thr = info["threshold"]

    X_test, y_test = xy(load_split("test"))
    X_val, y_val = xy(load_split("val"))
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= thr).astype(int)

    results = {
        "model": info["model"],
        "n_test": int(len(y_test)),
        "test_churn_rate": float(y_test.mean()),
        "selected": classification_metrics(y_test, proba, thr),
        "selected_at_0.5": classification_metrics(y_test, proba, 0.5),
        "dummy_baseline": classification_metrics(
            y_test, dummy.predict_proba(X_test)[:, 1], 0.5),
        "top_decile_lift": top_decile_lift(y_test, proba),
        "strategies": {
            "contact_nobody": 0.0,
            "contact_everyone": business_value(y_test, np.ones(len(y_test), dtype=int)),
            "use_model": business_value(y_test, pred),
        },
    }
    with open(config.REPORTS_DIR / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    s = results["selected"]
    print(f"Test results for {info['model']} (threshold {thr:.2f})")
    print(f"  ROC-AUC {s['roc_auc']:.3f} | PR-AUC {s['pr_auc']:.3f} "
          f"(baseline PR-AUC {results['dummy_baseline']['pr_auc']:.3f})")
    print(f"  Precision {s['precision']:.3f} | Recall {s['recall']:.3f} | F1 {s['f1']:.3f}")
    print(f"  Top-decile lift {results['top_decile_lift']:.2f}x")
    print(f"  Net business value on test set: {results['strategies']}")

    # ---- figures ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Stay", "Churn"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix (threshold {thr:.2f})")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[0], name=info["model"])
    axes[0].plot([0, 1], [0, 1], "k--", lw=1); axes[0].set_title("ROC curve (test)")
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[1], name=info["model"])
    axes[1].axhline(y_test.mean(), color="k", ls="--", lw=1, label="Baseline rate")
    axes[1].legend(); axes[1].set_title("Precision-recall curve (test)")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "roc_pr_curves.png", dpi=150); plt.close(fig)

    # threshold vs business value, drawn from VALIDATION (where it was chosen)
    val_proba = model.predict_proba(X_val)[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    vals = [business_value(y_val, (val_proba >= t).astype(int)) for t in grid]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, vals); ax.axvline(thr, color="red", ls="--", label=f"chosen = {thr:.2f}")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Net business value (validation)")
    ax.set_title("Choosing the threshold by business value"); ax.legend()
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "threshold_vs_value.png", dpi=150); plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(mean_pred, frac_pos, "o-", label=info["model"]); ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration (test)"); ax.legend()
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "calibration.png", dpi=150); plt.close(fig)
    print(f"Figures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
