# Fit-to-serve decision: `sale_probability` sale_probability-20260420-49514d06

**APPROVE** (2026-10-16T12:44:26; train 27,761 rows, test 2,726 rows)

| check | result | detail |
|---|---|---|
| pre_ship_review | PASS | verdict=pass_with_warnings |
| threshold | PASS | auc=0.7874 vs threshold 0.75 |
| beats_baseline | PASS | auc 0.7874 vs baseline 0.7435: 18.0% better above chance (0.5) (need 10%) |
| vs_champion | PASS | vs sale_probability-20251101-03e1e176: improvement -0.0001 auc (95% CI -0.0071 .. +0.0084); non-inferiority margin 0.01 |
| slices_vs_champion | PASS | no slice regressed beyond tolerance |
| package_reproduces | PASS | max |packaged - in-memory| = 0 over 2726 rows |
| serving_parity | PASS | 100.00% of 604 replayed requests match training features; 0 prediction mismatches |
| latency | PASS | p50 28.9 ms, p95 37.5 ms (budget 100.0 ms) |
