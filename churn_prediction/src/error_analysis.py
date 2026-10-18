"""Error analysis: where does the model fail, and what drives its predictions?

Outputs (in reports/):
  error_by_segment.csv     error / FN / FP rates for key customer segments
  worst_false_negatives.csv  churners the model was most confident would stay
  worst_false_positives.csv  stayers the model was most confident would churn
  feature_importance.csv + figures/feature_importance.png  (permutation importance)
  error_analysis.md        short written summary to paste into your report

Run:  python -m src.error_analysis
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config
from src.data_prep import load_split, xy
from src.features import add_features

SEGMENTS = ["Contract", "tenure_group", "InternetService", "PaymentMethod", "SeniorCitizen"]


def segment_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in SEGMENTS:
        for value, g in df.groupby(col, observed=True):
            pos, neg = (g["actual"] == 1).sum(), (g["actual"] == 0).sum()
            rows.append({
                "segment": col, "value": value, "customers": len(g), "churners": int(pos),
                "actual_churn_rate": g["actual"].mean(),
                "error_rate": (g["actual"] != g["predicted"]).mean(),
                "false_negative_rate": ((g["actual"] == 1) & (g["predicted"] == 0)).sum() / pos if pos else float("nan"),
                "false_positive_rate": ((g["actual"] == 0) & (g["predicted"] == 1)).sum() / neg if neg else float("nan"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(config.FINAL_MODEL_PATH)
    with open(config.THRESHOLD_PATH) as f:
        thr = json.load(f)["threshold"]

    test = load_split("test")
    X_test, y_test = xy(test)
    proba = model.predict_proba(X_test)[:, 1]

    df = add_features(X_test)
    df["actual"], df["proba"] = y_test.values, proba
    df["predicted"] = (proba >= thr).astype(int)

    seg = segment_table(df)
    seg.to_csv(config.REPORTS_DIR / "error_by_segment.csv", index=False)

    keep = ["Contract", "tenure", "MonthlyCharges", "InternetService", "PaymentMethod", "actual", "proba"]
    df[(df.actual == 1) & (df.predicted == 0)].sort_values("proba").head(20)[keep] \
        .to_csv(config.REPORTS_DIR / "worst_false_negatives.csv", index=False)
    df[(df.actual == 0) & (df.predicted == 1)].sort_values("proba", ascending=False).head(20)[keep] \
        .to_csv(config.REPORTS_DIR / "worst_false_positives.csv", index=False)

    # permutation importance on RAW columns (the pipeline accepts raw data)
    imp = permutation_importance(
        model, X_test, y_test, scoring="average_precision",
        n_repeats=5, random_state=config.SEED, n_jobs=1)
    imp_df = (pd.DataFrame({"feature": X_test.columns,
                            "importance": imp.importances_mean,
                            "std": imp.importances_std})
              .sort_values("importance", ascending=False))
    imp_df.to_csv(config.REPORTS_DIR / "feature_importance.csv", index=False)

    top = imp_df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#2b6cb0")
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.set_title("Top 10 features (permutation importance, test)")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "feature_importance.png", dpi=150); plt.close(fig)

    # ---- written summary ------------------------------------------------------
    big = seg[seg.customers >= 30]
    worst_err = big.sort_values("error_rate", ascending=False).head(3)
    # rates on tiny groups are noise, so require >= 20 actual churners
    worst_fn = seg[seg.churners >= 20].sort_values("false_negative_rate", ascending=False).head(3)
    n_fn = int(((df.actual == 1) & (df.predicted == 0)).sum())
    n_fp = int(((df.actual == 0) & (df.predicted == 1)).sum())
    lines = [
        "# Error analysis (test set)", "",
        f"- Threshold used: **{thr:.2f}**",
        f"- Missed churners (false negatives): **{n_fn}**; unnecessary offers (false positives): **{n_fp}**", "",
        "## Segments with the highest overall error rate (>= 30 customers)", "",
    ]
    lines += [f"- {r.segment} = {r.value}: error {r.error_rate:.1%} (n={r.customers})"
              for r in worst_err.itertuples()]
    lines += ["", "## Segments where most churners are missed (>= 20 churners)", ""]
    lines += [f"- {r.segment} = {r.value}: {r.false_negative_rate:.1%} of churners missed (n={r.churners})"
              for r in worst_fn.itertuples()]
    lines += ["", "## Most important features", ""]
    lines += [f"- {r.feature} ({r.importance:.4f})" for r in imp_df.head(5).itertuples()]
    lines += ["", "> Interpretation (write your own): why might these segments be harder? "
              "What data would help (e.g. support tickets, usage drops)?", ""]
    (config.REPORTS_DIR / "error_analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
