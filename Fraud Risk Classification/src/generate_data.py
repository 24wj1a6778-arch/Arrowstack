"""Generate a SIMULATED card-transaction dataset with ~1.2% fraud.

Why simulate? The handbook requires public, simulated or authorized data.
Real fraud data is confidential, and the public Kaggle credit-card file has
anonymised features (V1..V28) that cannot be explained to a business user.
Here every feature has a plain meaning, so explanations are meaningful.

Design:
  * fraud label is drawn first (about 1.2% of rows)
  * fraud rows get shifted feature distributions (bigger amounts vs the
    customer's norm, night time, online/international, new device, ...)
  * 25% of fraud is "stealth": it looks like normal behaviour, so no model
    can catch it from these features (realistic, and good for error analysis)
  * a few values are missing, as in real data

Run:  python -m src.generate_data
"""
import argparse

import numpy as np
import pandas as pd

from src import config

CATEGORIES = ["grocery", "restaurant", "fuel", "travel", "electronics",
              "online_retail", "entertainment", "health"]
P_LEGIT_CAT = [0.24, 0.18, 0.12, 0.06, 0.07, 0.17, 0.08, 0.08]
P_FRAUD_CAT = [0.05, 0.05, 0.05, 0.14, 0.27, 0.28, 0.12, 0.04]


def generate(n: int = 60000, fraud_rate: float = 0.012, seed: int = config.SEED,
             stealth_share: float = 0.25) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fraud = rng.random(n) < fraud_rate
    stealth = fraud & (rng.random(n) < stealth_share)
    shifted = fraud & ~stealth          # rows that actually look suspicious

    def pick(p_legit, p_fraud):
        return np.where(shifted, p_fraud, p_legit)

    # customer baseline
    avg_amount = np.exp(rng.normal(3.8, 0.6, n)).round(2)          # ~ $45 typical
    ratio = np.where(shifted, np.exp(rng.normal(1.1, 0.8, n)), np.exp(rng.normal(0.0, 0.5, n)))
    amount = (avg_amount * ratio).clip(1, 5000).round(2)

    hour_legit = np.mod(rng.normal(14, 5, n), 24)
    hour_fraud = np.where(rng.random(n) < 0.45, rng.uniform(0, 6, n), hour_legit)
    hour = np.where(shifted, hour_fraud, hour_legit).astype(int) % 24
    day = rng.integers(0, 7, n)

    channel = np.empty(n, dtype=object)
    u = rng.random(n)
    p_online = pick(0.35, 0.68)
    p_pos = pick(0.55, 0.18)
    channel[:] = np.where(u < p_online, "online", np.where(u < p_online + p_pos, "pos", "atm"))

    cat = np.empty(n, dtype=object)
    cat[~shifted] = rng.choice(CATEGORIES, size=(~shifted).sum(), p=P_LEGIT_CAT)
    cat[shifted] = rng.choice(CATEGORIES, size=shifted.sum(), p=P_FRAUD_CAT)

    intl = (rng.random(n) < pick(0.04, 0.28)).astype(int)
    distance = np.where(shifted, np.exp(rng.normal(3.6, 1.3, n)), np.exp(rng.normal(2.0, 1.0, n))).round(1)
    velocity = np.where(shifted, rng.poisson(3.5, n), rng.poisson(1.4, n))
    card_age = rng.integers(30, 2000, n)
    new_card = shifted & (rng.random(n) < 0.30)
    card_age = np.where(new_card, rng.integers(1, 45, n), card_age)
    account_age = card_age + rng.integers(0, 1500, n)
    new_merchant = (rng.random(n) < pick(0.12, 0.55)).astype(int)
    device_changed = (rng.random(n) < pick(0.03, 0.30)).astype(int)

    df = pd.DataFrame({
        "transaction_id": [f"TX{i:07d}" for i in range(n)],
        "amount": amount,
        "hour": hour,
        "day_of_week": day,
        "merchant_category": cat,
        "channel": channel,
        "is_international": intl,
        "distance_from_home_km": distance,
        "txn_count_24h": velocity,
        "avg_amount_30d": avg_amount,
        "card_age_days": card_age,
        "account_age_days": account_age,
        "is_new_merchant": new_merchant,
        "device_changed": device_changed,
        "is_fraud": fraud.astype(int),
    })

    # realistic missing data
    df["distance_from_home_km"] = df["distance_from_home_km"].astype(float)
    df.loc[rng.random(n) < 0.01, "distance_from_home_km"] = np.nan
    df.loc[rng.random(n) < 0.005, "merchant_category"] = np.nan
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=60000)
    parser.add_argument("--fraud-rate", type=float, default=0.012)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--force", action="store_true", help="overwrite existing file")
    args = parser.parse_args()

    if config.RAW_PATH.exists() and not args.force:
        print(f"{config.RAW_PATH} already exists - keeping it (use --force to overwrite).")
        return
    config.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.n, args.fraud_rate, args.seed)
    df.to_csv(config.RAW_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {config.RAW_PATH}  "
          f"(fraud rate {df[config.TARGET].mean():.2%}, {int(df[config.TARGET].sum())} frauds)")


if __name__ == "__main__":
    main()
