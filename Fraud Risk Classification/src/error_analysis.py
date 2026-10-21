"""Error analysis: which frauds slip through, and which legit transactions get flagged?

Outputs (in reports/): error_by_segment.csv, worst_false_negatives.csv,
worst_false_positives.csv, caught_vs_missed.csv, error_analysis.md

Run:  python -m src.error_analysis
"""
import json

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data_prep import load_split, xy
from src.features import add_features

MIN_FRAUDS = 15   # ignore miss-rates for segments with fewer frauds (noise)


def segment_table(df: pd.DataFrame, segments: list) -> pd.DataFrame:
    rows = []
    for col in segments:
        for value, g in df.groupby(col, observed=True):
            frauds = int((g.actual == 1).sum())
            legit = int((g.actual == 0).sum())
            caught = int(((g.actual == 1) & (g.pred == 1)).sum())
            missed_amt = g.loc[(g.actual == 1) & (g.pred == 0), "amount"].sum()
            rows.append({
                "segment": col, "value": value, "transactions": len(g), "frauds": frauds,
                "fraud_rate": frauds / len(g),
                "missed_fraud_rate": (frauds - caught) / frauds if frauds else np.nan,
                "false_alarms": int(((g.actual == 0) & (g.pred == 1)).sum()),
                "false_alarm_rate": ((g.actual == 0) & (g.pred == 1)).sum() / legit if legit else np.nan,
                "missed_fraud_amount": float(missed_amt),
            })
    return pd.DataFrame(rows)


def main() -> None:
    model = joblib.load(config.FINAL_MODEL_PATH)
    with open(config.THRESHOLD_PATH) as f:
        thr = json.load(f)["threshold"]
    X_test, y_test = xy(load_split("test"))
    proba = model.predict_proba(X_test)[:, 1]

    df = add_features(X_test)
    df["actual"], df["proba"] = y_test.values, proba
    df["pred"] = (proba >= thr).astype(int)
    df["amount_band"] = pd.qcut(df["amount"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"], duplicates="drop")
    df["time_of_day"] = np.where(df["is_night"] == 1, "night (0-5h)", "day/evening")

    segments = ["merchant_category", "channel", "is_international", "amount_band",
                "time_of_day", "device_changed"]
    seg = segment_table(df, segments)
    seg.to_csv(config.REPORTS_DIR / "error_by_segment.csv", index=False)

    keep = ["amount", "channel", "merchant_category", "hour", "amount_ratio",
            "distance_from_home_km", "device_changed", "actual", "proba"]
    fn = df[(df.actual == 1) & (df.pred == 0)]
    fp = df[(df.actual == 0) & (df.pred == 1)]
    fn.sort_values("proba").head(25)[keep].to_csv(config.REPORTS_DIR / "worst_false_negatives.csv", index=False)
    fp.sort_values("proba", ascending=False).head(25)[keep].to_csv(config.REPORTS_DIR / "worst_false_positives.csv", index=False)

    # how do caught and missed frauds differ?
    fraud = df[df.actual == 1].copy()
    fraud["outcome"] = np.where(fraud.pred == 1, "caught", "missed")
    cmp_cols = ["amount", "amount_ratio", "distance_from_home_km", "txn_count_24h",
                "is_night", "is_international", "device_changed", "is_new_merchant", "new_card"]
    cvm = fraud.groupby("outcome")[cmp_cols].mean().T.round(3)
    cvm.to_csv(config.REPORTS_DIR / "caught_vs_missed.csv")

    big = seg[seg.frauds >= MIN_FRAUDS]
    worst_miss = big.sort_values("missed_fraud_rate", ascending=False).head(3)
    worst_fa = seg[seg.false_alarms > 0].sort_values("false_alarm_rate", ascending=False).head(3)
    lost = fn["amount"].sum()
    total_fraud = df.loc[df.actual == 1, "amount"].sum()
    lines = [
        "# Error analysis (test set)", "",
        f"- Threshold: **{thr:.4f}**",
        f"- Missed frauds: **{len(fn)}** of {int((df.actual == 1).sum())} "
        f"(${lost:,.0f} of ${total_fraud:,.0f} fraud value)",
        f"- False alarms: **{len(fp)}** legitimate transactions flagged", "",
        f"## Segments with the highest miss rate (>= {MIN_FRAUDS} frauds)", "",
    ]
    lines += [f"- {r.segment} = {r.value}: {r.missed_fraud_rate:.0%} of frauds missed (n={r.frauds})"
              for r in worst_miss.itertuples()]
    lines += ["", "## Segments with the highest false-alarm rate", ""]
    lines += [f"- {r.segment} = {r.value}: {r.false_alarm_rate:.2%} of legit transactions flagged ({r.false_alarms})"
              for r in worst_fa.itertuples()]
    lines += ["", "## Caught vs missed frauds (average feature values)", "",
              "| feature | caught | missed |", "|---|---|---|"]
    for feat, r in cvm.iterrows():
        lines.append(f"| {feat} | {r.get('caught', float('nan')):.2f} | {r.get('missed', float('nan')):.2f} |")
    lines += ["", "> Interpretation (write your own): missed frauds look like normal customer behaviour "
              "(small amount vs usual, daytime, familiar device).  What extra data would expose them "
              "(device fingerprint, merchant history, session behaviour)?", ""]
    (config.REPORTS_DIR / "error_analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
