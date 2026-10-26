# Customer Churn Prediction 

Predict which telecom customers are likely to churn so a retention team can decide who gets an offer.
Covers every handbook requirement: train/validation/test discipline, baseline model, metric comparison,
error analysis, reproducible pipeline and model card.

> **Data note:** the project ships with a *simulated* dataset that mirrors the public IBM/Kaggle
> "Telco Customer Churn" schema. To use the real file, save it as `data/raw/telco_churn.csv` and skip the
> generate step - nothing else changes.

## Project structure

```
churn_prediction/
├── README.md                 <- you are here
├── requirements.txt
├── run_all.py                <- runs the whole pipeline with one command
├── data/
│   ├── raw/telco_churn.csv   <- raw data (simulated or real)
│   └── processed/            <- train.csv, val.csv, test.csv, sample_new_customers.csv
├── src/
│   ├── config.py             <- paths, seed, split sizes, feature lists, business assumptions
│   ├── generate_data.py      <- simulated Telco-style data generator
│   ├── data_prep.py          <- load, validate, clean, stratified 60/20/20 split
│   ├── eda.py                <- exploratory analysis (train split only)
│   ├── features.py           <- feature engineering + preprocessing pipeline
│   ├── metrics.py            <- classification + business-value metrics
│   ├── train.py              <- baseline + 3 models, validation comparison, threshold choice
│   ├── evaluate.py           <- one-time test evaluation + plots
│   ├── error_analysis.py     <- segment errors, worst cases, permutation importance
│   ├── model_card.py         <- builds reports/model_card.md from real results
│   └── predict.py            <- score new customers (with input validation)
├── tests/test_pipeline.py    <- 18 unit tests
├── models/                   <- churn_model.joblib, threshold.json, candidates/
├── reports/                  <- metrics, figures, model_card.md, error_analysis.md, eda_summary.md
└── docs/                     <- requirements, decision log, work log, presentation outline, viva prep
```

## How to run (step by step)

```bash
# 0. one-time setup (Python 3.10+)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Option A: everything in one go
python run_all.py

# Option B: step by step (recommended the first time, so you understand each stage)
python -m src.generate_data          # 1. create data/raw/telco_churn.csv
python -m src.data_prep              # 2. clean + split into train/val/test
python -m src.eda                    # 3. look at the data (reports/eda_summary.md + figures)
python -m src.train                  # 4. train + compare models, pick model + threshold
python -m src.evaluate               # 5. final test-set results + plots
python -m src.error_analysis         # 6. where does the model fail and why?
python -m src.model_card             # 7. write reports/model_card.md
python -m src.predict --input data/processed/sample_new_customers.csv --output reports/sample_predictions.csv

# Tests
python -m unittest discover -s tests -v
```

Optional extras: `python -m src.train --tune` (small hyper-parameter search),
`pip install mlflow` then `python -m src.train --mlflow` (experiment tracking, then run `mlflow ui`).

## Method in brief

1. **Split first.** Stratified 60/20/20 train/validation/test. The test set is used once, in `evaluate.py`.
2. **Baseline.** `DummyClassifier` predicting the base churn rate; every real model must beat it.
3. **Feature engineering.** tenure groups, average monthly spend, number of services, charge per service,
   month-to-month flag, autopay flag.
4. **One pipeline object.** Feature engineering + imputation + scaling + one-hot encoding + model are a single
   scikit-learn `Pipeline`, so training and prediction use identical logic.
5. **Model comparison.** Logistic regression, random forest, histogram gradient boosting, compared on the
   validation set. **PR-AUC** is the selection metric because only ~29% of customers churn
   (accuracy would be misleading).
6. **Business threshold.** The decision threshold maximises net value of retention offers
   (offer cost 20, acceptance 30%, saved customer worth 200 - *simulated assumptions in `src/config.py`*).
7. **Error analysis.** Error/FN/FP rates by segment, worst mistakes, permutation importance.

## Results (hold-out test set, simulated data, seed 42)

| Metric | Selected model (logistic regression) | Dummy baseline |
|---|---|---|
| PR-AUC | 0.615 | 0.286 |
| ROC-AUC | 0.814 | 0.500 |
| Precision / Recall | 0.592 / 0.613 | 0 / 0 |
| Top-decile lift | 2.52x | - |
| Net business value (test set) | **+6,480** vs -4,000 contacting everyone | 0 contacting nobody |

Details: `reports/model_card.md`, `reports/error_analysis.md`, `reports/figures/`.

**Honest caveat:** the simulated data is generated from a logistic process, so logistic regression naturally wins.
On the real Telco file the gap between models is usually small and the ranking can change - report whatever you find.

## Handbook acceptance criteria -> evidence

| Criterion | Where to find it |
|---|---|
| Core requirements complete and mapped to evidence | this table + `docs/requirements.md` |
| Reproducible by a mentor | `requirements.txt`, fixed seed, `run_all.py`, this README |
| Validation documented, known issues disclosed | `reports/model_comparison_val.csv`, `reports/test_metrics.json`, `reports/test_results.txt`, model card limitations |
| Professionally organised and consistently named | folder structure above |
| Can explain decisions, trade-offs, limitations | `docs/decision_log.md`, `docs/viva_prep.md` |

## Limitations

- Simulated data; real performance will differ.
- Business value depends on assumed costs and acceptance rate.
- Probabilities are only roughly calibrated.
- Correlation, not causation: a flagged customer may need a service fix, not an offer.
- Review fairness of `SeniorCitizen`/`gender` before any production use.

## Next improvements

Time-based validation, probability calibration, behavioural features (support tickets, usage drops),
SHAP explanations, drift monitoring, a small API or dashboard for the retention team.
