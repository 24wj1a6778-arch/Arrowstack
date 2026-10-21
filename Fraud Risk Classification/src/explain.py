"""Explainability: global importance + per-transaction explanations.

SHAP is optional in the handbook.  To keep the project installable everywhere
we use two model-agnostic methods that work through the full pipeline:

  GLOBAL  permutation importance - shuffle one raw column, see how much PR-AUC drops.
  LOCAL   "what-if" attribution  - for one transaction, replace a feature by
          typical values from the training data and measure how much the fraud
          score falls.  A big fall means that feature pushed the score up.

Local attributions are NOT additive like SHAP values; they answer the analyst's
question "which of this transaction's features made it look risky?".
(SHAP is a documented next step: pip install shap, then shap.Explainer on the model.)

Outputs: reports/feature_importance.csv, reports/local_explanations.md,
         reports/figures/feature_importance.png, reports/figures/local_explanation.png

Run:  python -m src.explain
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config
from src.data_prep import load_split, xy


def local_explanation(model, row: pd.Series, background: pd.DataFrame) -> pd.DataFrame:
    """What-if attribution for one transaction (one row of raw columns)."""
    cols = config.REQUIRED_COLUMNS
    row_df = pd.DataFrame([row[cols]])
    score = float(model.predict_proba(row_df)[:, 1][0])
    out = []
    for col in cols:
        replaced = pd.concat([row_df] * len(background), ignore_index=True)
        replaced[col] = background[col].values
        typical = float(model.predict_proba(replaced)[:, 1].mean())
        out.append({"feature": col, "value": row[col], "score_if_typical": typical,
                    "contribution": score - typical})
    df = pd.DataFrame(out).sort_values("contribution", ascending=False).reset_index(drop=True)
    df.attrs["score"] = score
    return df


def describe(df: pd.DataFrame, top: int = 4) -> str:
    parts = []
    for r in df.head(top).itertuples():
        if r.contribution <= 0.005:
            break
        v = f"{r.value:.2f}" if isinstance(r.value, (float, np.floating)) else str(r.value)
        parts.append(f"{r.feature}={v} (+{r.contribution:.2f})")
    return ", ".join(parts) if parts else "no single dominant feature"


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(config.FINAL_MODEL_PATH)
    with open(config.THRESHOLD_PATH) as f:
        thr = json.load(f)["threshold"]
    train = load_split("train")
    X_test, y_test = xy(load_split("test"))
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= thr).astype(int)

    # ---- global ---------------------------------------------------------------
    imp = permutation_importance(model, X_test, y_test, scoring="average_precision",
                                 n_repeats=5, random_state=config.SEED, n_jobs=1)
    imp_df = (pd.DataFrame({"feature": X_test.columns, "importance": imp.importances_mean,
                            "std": imp.importances_std})
              .sort_values("importance", ascending=False))
    imp_df.to_csv(config.REPORTS_DIR / "feature_importance.csv", index=False)
    top = imp_df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#2b6cb0")
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.set_title("Top 10 features (permutation importance, test)")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "feature_importance.png", dpi=150); plt.close(fig)

    # ---- local ----------------------------------------------------------------
    background = train.drop(columns=[config.TARGET]).sample(200, random_state=config.SEED)
    test_df = X_test.copy()
    test_df["actual"], test_df["proba"], test_df["pred"] = y_test.values, proba, pred

    cases = {
        "Caught fraud (highest score)": test_df[(test_df.actual == 1) & (test_df.pred == 1)].nlargest(2, "proba"),
        "False alarm (legit, flagged)": test_df[(test_df.actual == 0) & (test_df.pred == 1)].nlargest(1, "proba"),
        "Missed fraud (lowest score)": test_df[(test_df.actual == 1) & (test_df.pred == 0)].nsmallest(2, "proba"),
    }
    lines = ["# Local explanations (test set)", "",
             f"Threshold {thr:.4f}. 'Contribution' = how much the fraud score would drop if that "
             "feature took a typical value.", ""]
    first_flagged = None
    for title, rows in cases.items():
        lines += [f"## {title}", ""]
        for idx, row in rows.iterrows():
            expl = local_explanation(model, row, background)
            if first_flagged is None:
                first_flagged = (idx, expl)
            lines.append(f"- Transaction row {idx}: score **{expl.attrs['score']:.3f}**, "
                         f"amount {row['amount']:.2f}, channel {row['channel']}, "
                         f"category {row['merchant_category']}. Drivers: {describe(expl)}")
        lines.append("")
    lines += ["> Missed frauds usually show no strong driver: they look like normal behaviour "
              "(the 'stealth' fraud built into the simulation). Explain this in your presentation.", ""]
    (config.REPORTS_DIR / "local_explanations.md").write_text("\n".join(lines))

    idx, expl = first_flagged
    top5 = expl.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([f"{r.feature} = {r.value if not isinstance(r.value, float) else round(r.value, 1)}"
             for r in top5.itertuples()], top5["contribution"], color="#c53030")
    ax.set_xlabel("Fraud score drop if feature were typical")
    ax.set_title(f"Why was this transaction flagged? (score {expl.attrs['score']:.2f})")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "local_explanation.png", dpi=150); plt.close(fig)

    print(imp_df.head(8).round(4).to_string(index=False))
    print("\n" + "\n".join(lines[:14]))


if __name__ == "__main__":
    main()
