# Error analysis (test set)

- Threshold: **0.2754**
- Missed frauds: **56** of 150 ($3,314 of $24,504 fraud value)
- False alarms: **63** legitimate transactions flagged

## Segments with the highest miss rate (>= 15 frauds)

- amount_band = Q2: 87% of frauds missed (n=23)
- merchant_category = grocery: 69% of frauds missed (n=16)
- merchant_category = restaurant: 69% of frauds missed (n=16)

## Segments with the highest false-alarm rate

- time_of_day = night (0-5h): 2.76% of legit transactions flagged (24)
- is_international = 1: 2.25% of legit transactions flagged (11)
- merchant_category = electronics: 1.82% of legit transactions flagged (15)

## Caught vs missed frauds (average feature values)

| feature | caught | missed |
|---|---|---|
| amount | 225.42 | 59.19 |
| amount_ratio | 4.41 | 1.24 |
| distance_from_home_km | 85.22 | 32.28 |
| txn_count_24h | 3.34 | 1.43 |
| is_night | 0.52 | 0.05 |
| is_international | 0.27 | 0.05 |
| device_changed | 0.24 | 0.09 |
| is_new_merchant | 0.64 | 0.21 |
| new_card | 0.41 | 0.04 |

> Interpretation (write your own): missed frauds look like normal customer behaviour (small amount vs usual, daytime, familiar device).  What extra data would expose them (device fingerprint, merchant history, session behaviour)?
