<!-- Kept from the replay run before the champion margin was revised from 0.003 to 0.01 AUC; see README, Findings. -->

# Fit-to-serve decision: `sale_probability` sale_probability-20260427-81a78026

**REJECT** (2026-10-16T12:34:11; train 27,879 rows, test 2,765 rows)

| check | result | detail |
|---|---|---|
| pre_ship_review | PASS | verdict=pass_with_warnings |
| threshold | PASS | auc=0.7958 vs threshold 0.75 |
| beats_baseline | PASS | auc 0.7958 vs baseline 0.7602: 13.7% better above chance (0.5) (need 10%) |
| vs_champion | FAIL | vs sale_probability-20251101-55214d71: improvement -0.0010 auc (95% CI -0.0095 .. +0.0079); non-inferiority margin 0.003 |
| slices_vs_champion | PASS | no slice regressed beyond tolerance |
| package_reproduces | PASS | max |packaged - in-memory| = 0 over 2765 rows |
| serving_parity | PASS | 100.00% of 602 replayed requests match training features; 0 prediction mismatches |
| latency | PASS | p50 35.6 ms, p95 46.1 ms (budget 100.0 ms) |
