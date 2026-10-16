# Pre-ship review: `price_estimator_leaky`

**Verdict: block**

| check | severity | result | detail |
|---|---|---|---|
| temporal_split | error | FAIL | random split on time-ordered data: test rows sit between training rows, so the model is evaluated on the past it was trained around |
| label_availability | error | FAIL | 100.00% of training labels were only known after the first test prediction (2025-03-01) |
| post_event_features | error | FAIL | features not available at 'listing': ['auction_outcome.n_bids'] |
| point_in_time_join | error | FAIL | point_in_time disabled in spec; rows joined to snapshots taken after the prediction: auction_outcome=100.00%, model_market_90d=100.00%, segment_market_28d=100.00%, seller_90d=100.00% |
| backward_window | error | PASS | every aggregated value was known before prediction time |
| univariate_leakage_probe | warn | PASS | no single feature determines the label |
| missingness_shift | warn | PASS | missing-value rates consistent across splits |
| test_size | error | PASS | test rows=4837 (min 1000) |
| baseline_defined | warn | PASS | baseline=model_market_90d.median_price_90d |
| test_window | warn | PASS | test window spans 43.6 weeks |
| slice_coverage | info | PASS | every slice large enough to gate on |
