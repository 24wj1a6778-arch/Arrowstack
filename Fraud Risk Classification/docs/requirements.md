# Requirements Document - Fraud Risk Classification

## 1. Problem and stakeholder
- **Stakeholder:** fraud-operations manager at a card issuer.
- **Decision:** which transactions go to manual review (the team can review only a small share).
- **Problem:** about 1.25% of transactions are fraud. Missing fraud loses money; reviewing everything wastes analyst time and annoys customers.

## 2. Inputs and outputs
- **Input:** one row per transaction - amount, hour, day, merchant category, channel, international flag, distance from home, transactions in last 24h, customer's 30-day average amount, card age, account age, new-merchant flag, device-changed flag (13 fields).
- **Output:** fraud score (0-1), risk band (alert / watch / ok) and decision (review / approve / invalid_input).

## 3. Success metrics (defined before modelling)
| Metric | Target |
|---|---|
| PR-AUC on test | far above the base rate (0.0125); goal >= 0.50 |
| Recall at 50% precision | >= 0.50 |
| Alert volume | <= 2% of transactions |
| Total cost | well below "approve everything" and "review everything" |
| Reproducibility | one command, same numbers |

## 4. Scope and constraints
- **In scope:** batch scoring model, threshold/operating-point analysis, explanations, error analysis.
- **Out of scope:** real-time serving, auto-declining customers, adaptive adversaries, deployment.
- Only simulated data; no personal data. Fixed seed 42.

## 5. Edge cases handled
| Case | Handling |
|---|---|
| Extreme class imbalance | PR-AUC for selection, four imbalance strategies compared, resampling on training data only |
| Missing distance / merchant category | imputed inside the pipeline (median / most frequent) |
| Negative amount, hour outside 0-23, negative counts | row flagged `invalid_input`, never scored |
| Missing required column | clear error naming the column |
| Unseen merchant category | ignored by the encoder, no crash |
| Decision/band disagreement | both derived from the same threshold, covered by a test |
| Metric uncertainty (few frauds) | bootstrap confidence interval for PR-AUC |

## 6. Acceptance criteria
See the mapping table in `README.md`.

## 7. Assumptions (simulated; state them in the presentation)
A missed fraud costs the transaction amount; each alert costs 5 (review effort); caught fraud is fully blocked.
