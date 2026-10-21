# Presentation Outline (9-10 slides, ~10 minutes)

1. **Problem and stakeholder** - fraud costs money, but analysts can only review a few transactions.
2. **Why accuracy fails** - 98.75% accurate by saying "legit" every time; success metrics instead.
3. **Data** - 60,000 simulated transactions, 1.25% fraud, 25% stealth fraud; assumptions stated.
4. **What fraud looks like** - `eda_fraud_by_segment.png`, `eda_distributions.png` (large vs usual amount, new merchant, night, many transactions).
5. **Method** - stratified split, baseline, 3 models x 4 imbalance strategies, resampling only on training data.
6. **Model comparison** - table from `model_comparison_val.csv`; honest point: strategies are within noise.
7. **Threshold analysis** - `threshold_tradeoff.png`; alert budget table; cost-optimal point.
8. **Test results** - PR-AUC 0.648 (CI 0.58-0.72), precision 0.60, recall 0.63, cost 4,099 vs 24,504 / 60,000; `cost_comparison.png`.
9. **Explainability** - `feature_importance.png`, `local_explanation.png` (why this transaction was flagged).
10. **Errors, limitations, next steps** - stealth fraud missed, simulated data, few frauds, time-based validation, SHAP, monitoring; demo `predict.py`.

**Tip:** one takeaway sentence per slide, evidence underneath, code stays in the repo.
