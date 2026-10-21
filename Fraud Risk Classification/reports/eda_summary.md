# EDA summary (train split)

- Transactions: 36,000 | frauds: 449 | fraud rate **1.25%**
- A model that never flags anything is 98.8% accurate -> accuracy is useless; use PR-AUC, precision, recall.
- Missing values: {'merchant_category': 180, 'distance_from_home_km': 348}

- Fraud rate by **merchant_category**: restaurant=0.5%, grocery=0.5%, health=0.5%, fuel=0.7%, online_retail=1.8%, travel=2.1%, entertainment=2.5%, electronics=3.5%
- Fraud rate by **channel**: pos=0.7%, atm=1.3%, online=2.0%
- Fraud rate by **is_international**: 0=1.0%, 1=5.9%
- Fraud rate by **device_changed**: 0=1.0%, 1=8.4%
- Fraud rate by **is_night**: 0=0.8%, 1=6.1%
- Fraud rate by **is_new_merchant**: 0=0.8%, 1=4.5%

Median values (0 = legit, 1 = fraud):

```
          amount  amount_ratio  distance_from_home_km  txn_count_24h
is_fraud                                                            
0          44.77          0.97                    7.4            1.0
1          96.92          2.21                   22.6            3.0
```
