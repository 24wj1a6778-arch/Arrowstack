# Requirements Document - Customer Churn Prediction

## 1. Problem and stakeholder
- **Stakeholder:** customer-retention manager at a telecom company.
- **Decision:** each month, choose which customers get a retention offer.
- **Problem:** roughly 29% of customers churn; contacting everyone costs more than it saves, contacting no one loses customers.

## 2. Inputs and outputs
- **Input:** one row per customer - demographics, subscribed services, contract, billing, tenure, monthly/total charges (20 columns).
- **Output:** churn probability (0-1), risk band (low/medium/high) and a decision (`offer_retention` / `no_action` / `invalid_input`).

## 3. Success metrics (defined before modelling)
| Metric | Target |
|---|---|
| PR-AUC on test set | clearly above the base rate (0.29); goal >= 0.55 |
| Recall of churners at the chosen threshold | >= 0.55 |
| Precision at the chosen threshold | >= 0.50 |
| Net business value | positive and above "contact everyone" |
| Reproducibility | mentor can rerun with one command and get the same numbers |

## 4. Constraints and scope
- **In scope:** tabular churn model, evaluation, explanation, batch scoring script.
- **Out of scope:** real-time serving, causal uplift modelling, deployment.
- Only public/simulated data; no personal data.
- Fixed random seed (42) for reproducibility.

## 5. Edge cases handled
| Case | Handling |
|---|---|
| Blank `TotalCharges` (tenure 0) | converted to NaN, median-imputed |
| Missing required column at prediction time | clear `ValueError` naming the column |
| Negative tenure/charges | row flagged `invalid_input`, not scored |
| Unseen category value | ignored by the encoder, no crash |
| Class imbalance | PR-AUC for selection, class weights, threshold tuned on business value |
| Duplicates / bad target values | removed / rejected during cleaning |

## 6. Acceptance criteria
See the mapping table in `README.md`.

## 7. Assumptions (simulated, state them in the presentation)
Offer cost 20; acceptance rate 30%; value of a retained customer 200.
