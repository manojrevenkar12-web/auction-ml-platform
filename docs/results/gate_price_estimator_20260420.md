# Fit-to-serve decision: `price_estimator` price_estimator-20260330-d3378224

**APPROVE** (2026-10-16T12:43:16; train 19,285 rows, test 1,703 rows)

| check | result | detail |
|---|---|---|
| pre_ship_review | PASS | verdict=pass_with_warnings |
| threshold | PASS | mape=0.0946 vs threshold 0.1 |
| beats_baseline | PASS | mape 0.0946 vs baseline 0.6856: 86.2% better (need 30%) |
| vs_champion | PASS | vs price_estimator-20260309-e331dfe5: improvement +0.0229 mape (95% CI +0.0205 .. +0.0257); non-inferiority margin 0.002; underpowered: CI half-width 0.0026 exceeds the margin, so non-inferiority can only be shown by a clearly better challenger or a longer test window |
| slices_vs_champion | PASS | no slice regressed beyond tolerance |
| package_reproduces | PASS | max |packaged - in-memory| = 0 over 1703 rows |
| serving_parity | PASS | 100.00% of 380 replayed requests match training features; 0 prediction mismatches |
| latency | PASS | p50 33.4 ms, p95 43.5 ms (budget 100.0 ms) |
