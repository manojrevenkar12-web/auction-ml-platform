# Fit-to-serve decision: `transport_cost` transport_cost-20260504-884d4ea6

**APPROVE** (2026-10-16T12:45:21; train 19,025 rows, test 1,866 rows)

| check | result | detail |
|---|---|---|
| pre_ship_review | PASS | verdict=pass_with_warnings |
| threshold | PASS | mape=0.0824 vs threshold 0.12 |
| beats_baseline | PASS | mape 0.0824 vs baseline 0.1232: 33.1% better (need 15%) |
| vs_champion | PASS | vs transport_cost-20260330-8b14360a: improvement +0.0089 mape (95% CI +0.0067 .. +0.0116); non-inferiority margin 0.002; underpowered: CI half-width 0.0025 exceeds the margin, so non-inferiority can only be shown by a clearly better challenger or a longer test window |
| slices_vs_champion | PASS | no slice regressed beyond tolerance |
| package_reproduces | PASS | max |packaged - in-memory| = 0 over 1866 rows |
| serving_parity | PASS | 100.00% of 414 replayed requests match training features; 0 prediction mismatches |
| latency | PASS | p50 20.4 ms, p95 27.1 ms (budget 100.0 ms) |
