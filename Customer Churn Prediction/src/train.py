"""Train and compare candidate models on the TRAIN split, select on VALIDATION.

Steps
  1. Fit a dummy baseline + 3 real models (same preprocessing pipeline).
  2. Score every model on the validation set (PR-AUC is the selection metric
     because churners are the minority class).
  3. Choose the best model, then choose the decision threshold on the
     validation set by maximising the business-value metric.
  4. Save everything.  The TEST set is NOT touched here.

Run:  python -m src.train            (add --tune for a small hyper-parameter search)
      python -m src.train --mlflow   (optional experiment tracking)
"""
import argparse
import json

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src import config
from src.data_prep import load_split, xy
from src.features import build_pipeline
from src.metrics import best_threshold, classification_metrics


def candidate_models() -> dict:
    s = config.SEED
    return {
        "dummy_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=s),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=s),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=15, l2_regularization=1.0,
            random_state=s),
    }


SEARCH_SPACES = {
    "logistic_regression": {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3]},
    "random_forest": {"model__max_depth": [4, 6, 8, None],
                      "model__min_samples_leaf": [3, 5, 10, 20]},
    "hist_gradient_boosting": {"model__learning_rate": [0.03, 0.05, 0.1],
                               "model__max_leaf_nodes": [8, 15, 31],
                               "model__l2_regularization": [0.0, 1.0, 5.0]},
}


def fit_model(name, estimator, X_train, y_train, tune=False):
    pipe = build_pipeline(estimator)
    if tune and name in SEARCH_SPACES:
        search = RandomizedSearchCV(
            pipe, SEARCH_SPACES[name], n_iter=6, scoring="average_precision",
            cv=StratifiedKFold(3, shuffle=True, random_state=config.SEED),
            random_state=config.SEED, n_jobs=1)
        search.fit(X_train, y_train)          # CV happens inside TRAIN only
        print(f"    tuned {name}: {search.best_params_}")
        return search.best_estimator_
    return pipe.fit(X_train, y_train)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    X_train, y_train = xy(load_split("train"))
    X_val, y_val = xy(load_split("val"))
    print(f"Train {len(X_train):,} rows | Val {len(X_val):,} rows")

    mlflow = None
    if args.mlflow:
        try:
            import mlflow  # optional dependency
            mlflow.set_experiment("churn-prediction")
        except ImportError:
            print("mlflow not installed - continuing without tracking.")

    config.CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    rows, fitted = [], {}
    for name, est in candidate_models().items():
        print(f"  training {name} ...")
        model = fit_model(name, est, X_train, y_train, tune=args.tune)
        proba = model.predict_proba(X_val)[:, 1]
        m = classification_metrics(y_val, proba, threshold=0.5)
        rows.append({"model": name, **m})
        fitted[name] = model
        joblib.dump(model, config.CANDIDATES_DIR / f"{name}.joblib")
        if mlflow:
            with mlflow.start_run(run_name=name):
                mlflow.log_metrics({k: float(v) for k, v in m.items() if k not in ("tn", "fp", "fn", "tp")})

    comparison = pd.DataFrame(rows).set_index("model")
    comparison.to_csv(config.REPORTS_DIR / "model_comparison_val.csv")
    show = ["pr_auc", "roc_auc", "precision", "recall", "f1", "business_value"]
    print("\nValidation comparison (threshold 0.5):")
    print(comparison[show].round(3).to_string())

    real = comparison.drop(index="dummy_baseline")
    best_name = real["pr_auc"].idxmax()
    best_model = fitted[best_name]
    val_proba = best_model.predict_proba(X_val)[:, 1]
    threshold, value = best_threshold(y_val, val_proba)
    print(f"\nSelected model: {best_name}")
    print(f"Chosen threshold: {threshold:.2f}  (validation business value {value:,.0f})")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, config.FINAL_MODEL_PATH)
    with open(config.THRESHOLD_PATH, "w") as f:
        json.dump({"model": best_name, "threshold": threshold,
                   "val_business_value": value,
                   "assumptions": {"offer_cost": config.OFFER_COST,
                                   "save_value": config.SAVE_VALUE,
                                   "accept_rate": config.ACCEPT_RATE}}, f, indent=2)
    print(f"Saved {config.FINAL_MODEL_PATH.name} and {config.THRESHOLD_PATH.name}")


if __name__ == "__main__":
    main()
