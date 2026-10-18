"""Generate a SIMULATED telecom churn dataset with the same schema as the
public IBM / Kaggle "Telco Customer Churn" dataset.

Why simulate? The handbook requires authorized/public/simulated data only.
This keeps the project runnable offline. If you download the real Telco file,
save it as data/raw/telco_churn.csv and skip this step - the rest of the
pipeline works unchanged.

Run:  python -m src.generate_data
"""
import argparse

import numpy as np
import pandas as pd

from src import config


def generate(n: int = 7043, seed: int = config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.21, 0.24]
    )
    mean_tenure = np.select(
        [contract == "Month-to-month", contract == "One year"], [18, 40], default=55
    )
    tenure = np.clip(rng.normal(mean_tenure, 18), 1, 72).round().astype(int)

    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22])
    has_net = internet != "No"

    def yes_no(p, need_net=False):
        vals = np.where(rng.random(n) < p, "Yes", "No")
        if need_net:
            vals = np.where(has_net, vals, "No internet service")
        return vals

    phone = yes_no(0.90)
    multiple = np.where(phone == "Yes", yes_no(0.42), "No phone service")
    online_sec = yes_no(0.30, True)
    online_bak = yes_no(0.34, True)
    device_prot = yes_no(0.34, True)
    tech_sup = yes_no(0.30, True)
    stream_tv = yes_no(0.38, True)
    stream_mov = yes_no(0.39, True)

    senior = (rng.random(n) < 0.16).astype(int)
    paperless = yes_no(0.59)
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"],
        size=n, p=[0.34, 0.23, 0.22, 0.21],
    )

    monthly = (
        20
        + np.where(internet == "DSL", 25, 0)
        + np.where(internet == "Fiber optic", 55, 0)
        + 5 * (online_sec == "Yes") + 5 * (online_bak == "Yes")
        + 5 * (device_prot == "Yes") + 5 * (tech_sup == "Yes")
        + 8 * (stream_tv == "Yes") + 8 * (stream_mov == "Yes")
        + 4 * (multiple == "Yes")
        + rng.normal(0, 3, n)
    ).clip(18.25, 119).round(2)
    total = (monthly * tenure * rng.uniform(0.95, 1.05, n)).round(2)

    logit = (
        -1.35
        + 1.0 * (contract == "Month-to-month")
        - 1.1 * (contract == "Two year")
        - 0.025 * tenure
        + 0.85 * (internet == "Fiber optic")
        - 0.55 * (online_sec == "Yes")
        - 0.55 * (tech_sup == "Yes")
        + 0.45 * (payment == "Electronic check")
        + 0.30 * (paperless == "Yes")
        + 0.35 * senior
        + 0.012 * (monthly - 65)
        + rng.normal(0, 0.6, n)           # unexplained noise
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame({
        "customerID": [f"{rng.integers(1000, 9999)}-{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], size=n),
        "SeniorCitizen": senior,
        "Partner": yes_no(0.48),
        "Dependents": yes_no(0.30),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_bak,
        "DeviceProtection": device_prot,
        "TechSupport": tech_sup,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_mov,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": np.where(churn, "Yes", "No"),
    })

    # Mimic a real data-quality problem: brand-new customers (tenure 0) have a
    # blank TotalCharges string in the public Telco file.
    new_idx = rng.choice(n, size=11, replace=False)
    df["TotalCharges"] = df["TotalCharges"].astype(object)  # text column, as in the real file
    df.loc[new_idx, "tenure"] = 0
    df.loc[new_idx, "TotalCharges"] = " "
    df.loc[new_idx, "Churn"] = "No"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=7043)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--force", action="store_true", help="overwrite existing file")
    args = parser.parse_args()

    if config.RAW_PATH.exists() and not args.force:
        print(f"{config.RAW_PATH} already exists - keeping it (use --force to overwrite).")
        return
    config.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.n, args.seed)
    df.to_csv(config.RAW_PATH, index=False)
    rate = (df[config.TARGET] == "Yes").mean()
    print(f"Wrote {len(df):,} rows to {config.RAW_PATH}  (churn rate {rate:.1%})")


if __name__ == "__main__":
    main()
