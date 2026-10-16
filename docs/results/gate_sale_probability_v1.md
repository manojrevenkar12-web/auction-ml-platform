# Fit-to-serve decision: `sale_probability` sale_probability-20251101-03e1e176

**APPROVE** (2026-10-16T12:39:27; train 27,851 rows, test 6,778 rows)

| check | result | detail |
|---|---|---|
| pre_ship_review | PASS | verdict=pass |
| threshold | PASS | auc=0.7963 vs threshold 0.75 |
| beats_baseline | PASS | auc 0.7963 vs baseline 0.7447: 21.1% better above chance (0.5) (need 10%) |
| vs_champion | PASS | no champion in this lane: first deployment |
| slices_vs_threshold | PASS | all slices within 0.7125 |
| package_reproduces | PASS | max |packaged - in-memory| = 0 over 6778 rows |
| serving_parity | PASS | 100.00% of 527 replayed requests match training features; 0 prediction mismatches |
| latency | PASS | p50 32.1 ms, p95 45.9 ms (budget 100.0 ms) |
