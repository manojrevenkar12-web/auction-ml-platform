# Fit-to-serve decision: `transport_cost` transport_cost-20260330-8b14360a

**APPROVE** (2026-10-16T12:43:28; train 18,907 rows, test 1,669 rows)

| check | result | detail |
|---|---|---|
| pre_ship_review | PASS | verdict=pass_with_warnings |
| threshold | PASS | mape=0.0926 vs threshold 0.12 |
| beats_baseline | PASS | mape 0.0926 vs baseline 0.1345: 31.1% better (need 15%) |
| vs_champion | PASS | vs transport_cost-20251101-57f9b623: improvement +0.0163 mape (95% CI +0.0135 .. +0.0195); non-inferiority margin 0.002; underpowered: CI half-width 0.0030 exceeds the margin, so non-inferiority can only be shown by a clearly better challenger or a longer test window |
| slices_vs_champion | PASS | no slice regressed beyond tolerance |
| package_reproduces | PASS | max |packaged - in-memory| = 0 over 1669 rows |
| serving_parity | PASS | 100.00% of 359 replayed requests match training features; 0 prediction mismatches |
| latency | PASS | p50 21.4 ms, p95 32.6 ms (budget 100.0 ms) |
