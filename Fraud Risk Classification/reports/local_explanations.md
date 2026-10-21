# Local explanations (test set)

Threshold 0.2754. 'Contribution' = how much the fraud score would drop if that feature took a typical value.

## Caught fraud (highest score)

- Transaction row 11930: score **0.995**, amount 459.68, channel online, category electronics. Drivers: txn_count_24h=4 (+0.14), merchant_category=electronics (+0.11), amount=459.68 (+0.11), is_new_merchant=1 (+0.03)
- Transaction row 6182: score **0.993**, amount 90.88, channel online, category online_retail. Drivers: txn_count_24h=7 (+0.15), merchant_category=online_retail (+0.12), distance_from_home_km=118.90 (+0.12), amount=90.88 (+0.10)

## False alarm (legit, flagged)

- Transaction row 2714: score **0.582**, amount 116.97, channel pos, category travel. Drivers: amount=116.97 (+0.44), avg_amount_30d=24.56 (+0.38), is_new_merchant=1 (+0.34), merchant_category=travel (+0.13)

## Missed fraud (lowest score)

- Transaction row 7448: score **0.000**, amount 20.46, channel pos, category restaurant. Drivers: no single dominant feature
- Transaction row 7660: score **0.000**, amount 69.92, channel atm, category restaurant. Drivers: no single dominant feature

> Missed frauds usually show no strong driver: they look like normal behaviour (the 'stealth' fraud built into the simulation). Explain this in your presentation.
