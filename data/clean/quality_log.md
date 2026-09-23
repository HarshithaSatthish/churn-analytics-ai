# Data Quality Log

- Raw rows: 7,043; raw columns: 21
- Clean rows: 7,043; rows dropped: 0
- Duplicate customer IDs: 0
- Blank `TotalCharges` fixed: 11 (all tenure = 0) -> 0.0
- Model-feature missing values after cleaning: 0
- Overall churn rate: 0.2654 (26.54%)
- Engineered for analysis: `tenure_band`, `num_services`, `is_fiber`, `is_month_to_month`, `is_electronic_check`, `churn_binary`
- Modeling note: exact duplicate indicator features are not used by the model.
