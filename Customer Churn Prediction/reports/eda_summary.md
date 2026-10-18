# EDA summary (train split)

- Rows: 4,225 | Overall churn rate: **28.6%** (class imbalance -> use PR-AUC, not accuracy)
- Missing values: {'TotalCharges': 3, 'avg_monthly_spend': 3}

- Churn by **Contract**: Two year=5%, One year=18%, Month-to-month=42%
- Churn by **InternetService**: DSL=20%, No=20%, Fiber optic=40%
- Churn by **PaymentMethod**: Credit card (automatic)=24%, Mailed check=26%, Bank transfer (automatic)=30%, Electronic check=32%
- Churn by **tenure_group**: 49-72m=8%, 25-48m=26%, 13-24m=37%, 7-12m=44%, 0-6m=46%
- Churn by **TechSupport**: No internet service=20%, Yes=27%, No=33%
- Churn by **SeniorCitizen**: 0=28%, 1=33%
