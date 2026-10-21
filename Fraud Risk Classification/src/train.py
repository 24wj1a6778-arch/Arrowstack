"""Train and compare models x imbalance strategies on TRAIN, select on VALIDATION.

Grid: 3 algorithms x 4 imbalance strategies (none / class_weight / undersample /
smote) + a dummy baseline = 13 models, all inside the same preprocessing pipeline.

Selection metric: validation PR-AUC (right metric for rare fraud).
Then the alert threshold is chosen on validation by minimising total cost.
The TEST set is not touched here.

Run:  python -m src.train
"""
import json
import time

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import config
from src.data_prep import load_split, xy
from src.features import build_pipeline
from src.metrics import best_threshold_cost, classification_metrics
from src.resampling import STRATEGIES, ResampledClassifier


def base_estimators() -> dict:
    s = config.SEED
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=s),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=s),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=200, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=s),
    }


def make_model(algorithm: str, strategy: str):
    """Build one pipeline for an (algorithm, imbalance strategy) pair."""
    est = clone(base_estimators()[algorithm])
    if strategy == "class_weight":
        est.set_params(class_weight="balanced_subsample" if algorithm == "random_forest" else "balanced")
    return build_pipeline(ResampledClassifier(est, strategy=strategy, random_state=config.SEED))


def dummy_model():
    return build_pipeline(DummyClassifier(strategy="prior"))


def main() -> None:
    X_train, y_train = xy(load_split("train"))
    X_val, y_val = xy(load_split("val"))
    amount_val = X_val[config.AMOUNT_COL].values
    print(f"Train {len(X_train):,} rows ({int(y_train.sum())} frauds) | "
          f"Val {len(X_val):,} rows ({int(y_val.sum())} frauds)")

    rows, fitted = [], {}
    combos = [("dummy_baseline", "none")] + [
        (a, s) for a in base_estimators() for s in STRATEGIES]
    for algorithm, strategy in combos:
        name = f"{algorithm}|{strategy}"
        t0 = time.time()
        model = dummy_model() if algorithm == "dummy_baseline" else make_model(algorithm, strategy)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        thr, cost = best_threshold_cost(y_val, proba, amount_val)
        m = classification_metrics(y_val, proba, thr, amount_val)
        rows.append({"algorithm": algorithm, "strategy": strategy, "pr_auc": m["pr_auc"],
                     "roc_auc": m["roc_auc"], "recall_at_precision_50": m["recall_at_precision_50"],
                     "best_threshold": thr, "precision": m["precision"], "recall": m["recall"],
                     "alerts_per_1000": m["alerts_per_1000"], "val_cost": m["total_cost"]})
        fitted[name] = (model, thr)
        print(f"  {name:38s} PR-AUC {m['pr_auc']:.3f}  ({time.time() - t0:4.1f}s)")

    comp = pd.DataFrame(rows)
    comp.to_csv(config.REPORTS_DIR / "model_comparison_val.csv", index=False)
    show = ["algorithm", "strategy", "pr_auc", "roc_auc", "recall_at_precision_50", "val_cost"]
    print("\nValidation comparison (each model at its own cost-optimal threshold):")
    print(comp[show].round(3).to_string(index=False))

    real = comp[comp.algorithm != "dummy_baseline"]
    best = real.loc[real["pr_auc"].idxmax()]
    best_name = f"{best.algorithm}|{best.strategy}"
    model, thr = fitted[best_name]
    print(f"\nSelected: {best.algorithm} with '{best.strategy}'  (val PR-AUC {best.pr_auc:.3f})")
    print(f"Chosen alert threshold: {thr:.4f}  (validation total cost {best.val_cost:,.0f})")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.FINAL_MODEL_PATH)
    with open(config.THRESHOLD_PATH, "w") as f:
        json.dump({"algorithm": best.algorithm, "strategy": best.strategy,
                   "threshold": thr, "val_pr_auc": float(best.pr_auc),
                   "val_cost": float(best.val_cost),
                   "assumptions": {"review_cost": config.REVIEW_COST,
                                   "missed_fraud_cost": "transaction amount"}}, f, indent=2)
    print(f"Saved {config.FINAL_MODEL_PATH.name} and {config.THRESHOLD_PATH.name}")


if __name__ == "__main__":
    main()
