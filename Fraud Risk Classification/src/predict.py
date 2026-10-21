"""Score new transactions with the saved model.

Run:
  python -m src.predict --input data/processed/sample_new_transactions.csv \
                        --output reports/sample_predictions.csv

Decision rule (consistent by design - decision and risk band always agree):
  score >= threshold        -> band "alert", decision "review"
  score >= threshold / 2    -> band "watch", decision "approve"
  otherwise                 -> band "ok",    decision "approve"
  invalid row               -> band "n/a",   decision "invalid_input" (never scored)

Edge cases handled
  * missing required column      -> clear error naming it
  * negative amount, hour outside 0-23, negative counts -> flagged invalid, not scored
  * missing / blank values       -> imputed by the pipeline
  * unseen category (e.g. new merchant type) -> ignored by the encoder, no crash
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src import config


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    for col in config.RAW_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in config.RAW_CATEGORICAL:
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) and v.strip() else None)
    return df


def invalid_mask(df: pd.DataFrame) -> pd.Series:
    bad = (df["amount"] < 0) | (df["hour"] < 0) | (df["hour"] > 23) | (df["txn_count_24h"] < 0)
    # an amount we cannot read at all cannot be scored sensibly either
    return (bad | df["amount"].isna()).fillna(True)


def predict_df(df: pd.DataFrame, model=None, threshold: float | None = None) -> pd.DataFrame:
    if model is None:
        model = joblib.load(config.FINAL_MODEL_PATH)
    if threshold is None:
        with open(config.THRESHOLD_PATH) as f:
            threshold = json.load(f)["threshold"]

    df = prepare_input(df)
    bad = invalid_mask(df)
    out = pd.DataFrame(index=df.index)
    out["fraud_score"] = np.nan
    out["risk_band"] = "n/a"
    out["decision"] = "invalid_input"
    ok = ~bad
    if ok.any():
        score = model.predict_proba(df.loc[ok, config.REQUIRED_COLUMNS])[:, 1]
        out.loc[ok, "fraud_score"] = score.round(4)
        out.loc[ok, "risk_band"] = np.where(score >= threshold, "alert",
                                            np.where(score >= threshold / 2, "watch", "ok"))
        out.loc[ok, "decision"] = np.where(score >= threshold, "review", "approve")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score transactions for fraud risk")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=str(config.REPORTS_DIR / "predictions.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = predict_df(df)
    final = pd.concat([df.reset_index(drop=True), result.reset_index(drop=True)], axis=1)
    final.to_csv(args.output, index=False)
    cols = ["amount", "channel", "merchant_category", "hour", "fraud_score", "risk_band", "decision"]
    print(final.sort_values("fraud_score", ascending=False)[cols].head(10).to_string(index=False))
    print(f"\nScored {len(final)} transactions -> {args.output}")
    print(final["decision"].value_counts().to_string())


if __name__ == "__main__":
    main()
