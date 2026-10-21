# Threshold analysis (validation set)

- Model: random_forest (smote)
- Cost-optimal threshold: **0.2754** -> 13.6 alerts per 1,000 transactions, precision 0.62, recall 0.67

How to read it: the more transactions the team reviews, the more fraud is caught (recall up) but the more reviews are wasted (precision down). The cost-optimal point is where one more review stops paying for itself.

See `operating_points_val.csv` for every operating point and `figures/threshold_tradeoff.png` for the curves.

> Write your own recommendation: which operating point would you give the fraud manager if analysts could review at most 1% of transactions?
