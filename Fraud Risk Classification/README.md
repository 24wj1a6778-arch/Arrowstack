# Fraud Risk Classification - Arrowstack ML Internship, Project 2

A simulated card-transaction fraud classifier with **imbalance handling, threshold analysis,
precision/recall trade-offs and explainability**. It covers every handbook requirement:
train/validation/test discipline, baseline, metric comparison, error analysis, reproducible
pipeline and model card.

> **Data note:** transactions are *simulated*  because real fraud data is
> confidential and public fraud sets use anonymised features that cannot be explained. About 1.25%
> of transactions are fraud, and 25% of that fraud is deliberately "stealth".

## Project structure

```
fraud_detection/
├── README.md, requirements.txt, run_all.py
├── data/raw/transactions.csv          <- simulated transactions
├── data/processed/                    <- train / val / test + sample_new_transactions.csv
├── src/
│   ├── config.py              settings, seed, cost assumptions
│   ├── generate_data.py       simulated data (60,000 rows)
│   ├── data_prep.py           validate, clean, stratified 60/20/20 split
│   ├── eda.py                 exploratory analysis (train only)
│   ├── features.py            feature engineering + one sklearn Pipeline
│   ├── resampling.py          undersampling + SMOTE (own, tested implementation)
│   ├── metrics.py             PR-AUC, recall@precision, alert rate, cost
│   ├── train.py               3 algorithms x 4 imbalance strategies + baseline
│   ├── threshold_analysis.py  precision/recall/cost across alert budgets
│   ├── evaluate.py            one-time test evaluation + bootstrap interval
│   ├── explain.py             global + per-transaction explanations
│   ├── error_analysis.py      which frauds are missed, which alerts are false
│   ├── model_card.py          builds reports/model_card.md from real results
│   └── predict.py             score new transactions (with input validation)
├── tests/test_pipeline.py     27 unit tests
├── models/                    fraud_model.joblib, threshold.json
├── reports/                   metrics, figures, model card, analyses
└── docs/                      requirements, decision log, work log, presentation outline, viva prep
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_all.py                    # everything in one go

# or step by step
python -m src.generate_data --force  # 1. simulate transactions
python -m src.data_prep              # 2. clean + split
python -m src.eda                    # 3. explore (train only)
python -m src.train                  # 4. compare 13 models, pick one + threshold
python -m src.threshold_analysis     # 5. precision/recall/cost operating points
python -m src.evaluate               # 6. final test-set results
python -m src.explain                # 7. what drives the scores
python -m src.error_analysis         # 8. where it fails
python -m src.model_card             # 9. model card
python -m src.predict --input data/processed/sample_new_transactions.csv --output reports/sample_predictions.csv

python -m unittest discover -s tests -v
```

## Method in brief

1. **Split first.** Stratified 60/20/20; resampling is applied to training data only, inside the model's `fit`.
2. **Baseline.** A dummy model predicting the base rate (PR-AUC about 0.013).
3. **Imbalance strategies compared fairly:** none, class weights, random undersampling, SMOTE, for
   logistic regression, random forest and gradient boosting.
4. **Metrics for rare events:** PR-AUC (selection), recall at 50% precision, alerts per 1,000, total cost.
   Accuracy is not used (always saying "legit" is 98.75% accurate).
5. **Threshold analysis:** operating points by alert budget, and the cost-optimal threshold
   (a missed fraud costs its amount; each alert costs 5, both *assumptions*).
6. **Explainability:** permutation importance (global) and what-if attributions (per transaction).
7. **Error analysis:** miss rate and false-alarm rate by segment, caught vs missed fraud.

## Results (hold-out test set: 12,000 transactions, 150 frauds, seed 42)

| Metric | Selected: random forest + SMOTE | Dummy baseline |
|---|---|---|
| PR-AUC | 0.648 (95% CI 0.578-0.719) | 0.013 |
| ROC-AUC | 0.853 | 0.500 |
| Precision / recall | 0.599 / 0.627 | 0 / 0 |
| Alerts per 1,000 transactions | 13.1 | 0 |
| Total cost | **4,099** | 24,504 (approve all), 60,000 (review all) |
| Fraud value caught | 21,189 of 24,504 | 0 |

Details: `reports/model_card.md`, `reports/threshold_analysis.md`, `reports/error_analysis.md`,
`reports/local_explanations.md`, `reports/figures/`.

**Honest findings**
- Every model is far better than the baseline, but the **imbalance strategies differ only slightly**
  (validation PR-AUC 0.615-0.673). With 150 validation frauds that gap is within noise, so do not claim
  that SMOTE is "the best technique". The selection is documented, not celebrated.
- The model catches 63% of frauds but **86% of fraud value**, because large, unusual transactions are the easiest to spot.
- Missed frauds look normal (small, daytime, familiar device): the simulated "stealth" fraud no feature can reveal.

## Handbook acceptance criteria -> evidence

| Criterion | Where to find it |
|---|---|
| Core requirements complete and mapped to evidence | this table + `docs/requirements.md` |
| Reproducible by a mentor | `requirements.txt`, fixed seed, `run_all.py`, this README |
| Validation documented, known issues disclosed | `reports/model_comparison_val.csv`, `test_metrics.json`, `test_results.txt`, model card limitations |
| Professionally organised | folder structure above |
| Can explain decisions and trade-offs | `docs/decision_log.md`, `docs/viva_prep.md` |

## Limitations

- Simulated data; real fraud is adversarial and drifts over time.
- Only 150 test frauds: metrics are noisy (see the confidence interval).
- Cost assumptions are invented; use the company's real numbers.
- Random split, not time-based; production should validate on later periods.
- Resampled models give scores, not calibrated probabilities.
- SMOTE here interpolates one-hot columns too (not SMOTE-NC); acceptable for ranking, noted as a limitation.

## Next improvements

Time-based validation, SHAP explanations (`pip install shap`), behavioural/session features,
probability calibration, analyst-feedback loop, per-segment thresholds, drift monitoring.
