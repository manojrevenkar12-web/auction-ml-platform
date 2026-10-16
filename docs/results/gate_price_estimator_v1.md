# Fit-to-serve decision: `price_estimator` price_estimator-20251101-302f7d7e

**APPROVE** (2026-10-16T12:39:07; train 19,486 rows, test 4,668 rows)

| check | result | detail |
|---|---|---|
| pre_ship_review | PASS | verdict=pass |
| threshold | PASS | mape=0.0795 vs threshold 0.1 |
| beats_baseline | PASS | mape 0.0795 vs baseline 0.6271: 87.3% better (need 30%) |
| vs_champion | PASS | no champion in this lane: first deployment |
| slices_vs_threshold | PASS | all slices within 0.1050 |
| package_reproduces | PASS | max |packaged - in-memory| = 0 over 4668 rows |
| serving_parity | PASS | 100.00% of 369 replayed requests match training features; 0 prediction mismatches |
| latency | PASS | p50 35.8 ms, p95 55.1 ms (budget 100.0 ms) |
