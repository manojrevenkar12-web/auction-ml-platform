# Fit-to-serve decision: `price_estimator` price_estimator-20260309-e331dfe5

**APPROVE** (2026-10-16T12:42:23; train 19,248 rows, test 1,748 rows)

| check | result | detail |
|---|---|---|
| pre_ship_review | PASS | verdict=pass_with_warnings |
| threshold | PASS | mape=0.0962 vs threshold 0.1 |
| beats_baseline | PASS | mape 0.0962 vs baseline 0.6406: 85.0% better (need 30%) |
| vs_champion | PASS | vs price_estimator-20260223-8fff1a06: improvement +0.0018 mape (95% CI +0.0008 .. +0.0029); non-inferiority margin 0.002 |
| slices_vs_champion | PASS | no slice regressed beyond tolerance |
| package_reproduces | PASS | max |packaged - in-memory| = 0 over 1748 rows |
| serving_parity | PASS | 100.00% of 371 replayed requests match training features; 0 prediction mismatches |
| latency | PASS | p50 32.8 ms, p95 44.0 ms (budget 100.0 ms) |
