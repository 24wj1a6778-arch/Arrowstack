# Error analysis (test set)

- Threshold used: **0.62**
- Missed churners (false negatives): **156**; unnecessary offers (false positives): **170**

## Segments with the highest overall error rate (>= 30 customers)

- tenure_group = 0-6m: error 36.7% (n=207)
- tenure_group = 7-12m: error 36.3% (n=124)
- Contract = Month-to-month: error 33.4% (n=781)

## Segments where most churners are missed (>= 20 churners)

- tenure_group = 49-72m: 85.2% of churners missed (n=27)
- InternetService = No: 78.1% of churners missed (n=64)
- Contract = One year: 77.6% of churners missed (n=58)

## Most important features

- tenure (0.2793)
- Contract (0.1243)
- TotalCharges (0.1045)
- InternetService (0.0418)
- TechSupport (0.0251)

> Interpretation (write your own): why might these segments be harder? What data would help (e.g. support tickets, usage drops)?
