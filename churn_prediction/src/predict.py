"""Score new customers with the saved model.

Run:
  python -m src.predict --input data/processed/sample_new_customers.csv \
                        --output reports/sample_predictions.csv

Input CSV must contain the raw feature columns (no Churn column needed).
Edge cases handled:
  * missing required column      -> clear error message
  * blank / non-numeric TotalCharges -> treated as missing and imputed
  * negative tenure / charges    -> row flagged "invalid_input" and not scored
  * unseen category values       -> ignored by the encoder (no crash)
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
        df[col] = pd.to_numeric(df[col].astype(str).str.strip().replace("", np.nan), errors="coerce")
    return df


def predict_df(df: pd.DataFrame, model=None, threshold: float | None = None) -> pd.DataFrame:
    if model is None:
        model = joblib.load(config.FINAL_MODEL_PATH)
    if threshold is None:
        with open(config.THRESHOLD_PATH) as f:
            threshold = json.load(f)["threshold"]

    df = prepare_input(df)
    invalid = (df["tenure"] < 0) | (df["MonthlyCharges"] < 0) | (df["TotalCharges"] < 0)
    out = pd.DataFrame(index=df.index)
    out["churn_probability"] = np.nan
    out["decision"] = "invalid_input"
    ok = ~invalid.fillna(False)
    if ok.any():
        proba = model.predict_proba(df.loc[ok, config.REQUIRED_COLUMNS])[:, 1]
        out.loc[ok, "churn_probability"] = proba.round(4)
        out.loc[ok, "decision"] = np.where(proba >= threshold, "offer_retention", "no_action")
    out["risk_band"] = pd.cut(out["churn_probability"], [-0.01, 0.2, 0.5, 1.0],
                              labels=["low", "medium", "high"]).astype(str)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score customers for churn risk")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=str(config.REPORTS_DIR / "predictions.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = predict_df(df)
    final = pd.concat([df.reset_index(drop=True), result.reset_index(drop=True)], axis=1)
    final.to_csv(args.output, index=False)
    print(final[["tenure", "Contract", "churn_probability", "risk_band", "decision"]].head(10).to_string())
    print(f"\nScored {len(final)} customers -> {args.output}")
    print(final["decision"].value_counts().to_string())


if __name__ == "__main__":
    main()
