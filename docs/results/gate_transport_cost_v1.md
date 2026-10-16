# Fit-to-serve decision: `transport_cost` transport_cost-20251101-57f9b623

**APPROVE** (2026-10-16T12:39:38; train 19,155 rows, test 4,629 rows)

| check | result | detail |
|---|---|---|
| pre_ship_review | PASS | verdict=pass |
| threshold | PASS | mape=0.0808 vs threshold 0.12 |
| beats_baseline | PASS | mape 0.0808 vs baseline 0.1232: 34.4% better (need 15%) |
| vs_champion | PASS | no champion in this lane: first deployment |
| slices_vs_threshold | PASS | all slices within 0.1260 |
| package_reproduces | PASS | max |packaged - in-memory| = 0 over 4629 rows |
| serving_parity | PASS | 100.00% of 365 replayed requests match training features; 0 prediction mismatches |
| latency | PASS | p50 20.4 ms, p95 27.8 ms (budget 100.0 ms) |
