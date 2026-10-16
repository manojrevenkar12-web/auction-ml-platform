# Pre-ship review: `sale_probability`

**Verdict: pass**

| check | severity | result | detail |
|---|---|---|---|
| temporal_split | error | PASS | time-based split |
| label_availability | error | PASS | 0.00% of training labels were only known after the first test prediction (2025-11-01) |
| post_event_features | error | PASS | all features exist at prediction time |
| point_in_time_join | error | PASS | every joined snapshot predates its prediction |
| backward_window | error | PASS | every aggregated value was known before prediction time |
| univariate_leakage_probe | warn | PASS | no single feature determines the label |
| missingness_shift | warn | PASS | missing-value rates consistent across splits |
| test_size | error | PASS | test rows=6778 (min 1000) |
| baseline_defined | warn | PASS | baseline=seller_90d.sell_through_90d |
| test_window | warn | PASS | test window spans 8.6 weeks |
| slice_coverage | info | PASS | every slice large enough to gate on |
