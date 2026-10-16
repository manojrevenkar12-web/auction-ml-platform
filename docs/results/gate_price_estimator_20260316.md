# Fit-to-serve decision: `price_estimator` price_estimator-20260223-8fff1a06

**APPROVE** (2026-10-16T12:41:43; train 19,305 rows, test 1,633 rows)

| check | result | detail |
|---|---|---|
| pre_ship_review | PASS | verdict=pass_with_warnings |
| threshold | PASS | mape=0.0761 vs threshold 0.1 |
| beats_baseline | PASS | mape 0.0761 vs baseline 0.6181: 87.7% better (need 30%) |
| vs_champion | PASS | vs price_estimator-20251101-302f7d7e: improvement +0.0127 mape (95% CI +0.0106 .. +0.0145); non-inferiority margin 0.002 |
| slices_vs_champion | PASS | no slice regressed beyond tolerance |
| package_reproduces | PASS | max |packaged - in-memory| = 0 over 1633 rows |
| serving_parity | PASS | 100.00% of 332 replayed requests match training features; 0 prediction mismatches |
| latency | PASS | p50 31.4 ms, p95 39.3 ms (budget 100.0 ms) |
