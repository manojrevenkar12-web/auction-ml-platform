# On-call runbook

Alerts are grouped into incidents per model (`aucml.monitoring.monitor.Incidents`):
you are paged when an incident opens or escalates from warn to critical, not
every week it stays open. Each alert names the model, version, check and subject.

## `performance` — delayed-label metric degraded

Scored weekly on last week's traffic, once labels have arrived, against the
metric the serving version had at release (`manifest.monitoring.reference_performance`),
overall and per slice. warn at +25% relative, critical at +50%.

1. Is it one slice or everything? A single slice (e.g. `fuel=ev`) usually means
   the market moved for that segment; everything moving usually means an
   upstream data or feature problem. Check `drift` and `data_quality` alerts for
   the same week before retraining.
2. Check label completeness (`labelled` column): a low count in a week can make
   a slice metric noisy.
3. If it is a real market move: retrain on a rolling window
   (`aucml release <model> --lane staging` with a recent cutoff). The gate compares
   the challenger with the prod champion on the same recent rows; promote through
   the `promote` workflow only if it approves.
4. If it is a data problem: fix upstream, do not retrain on broken data.

Note from the replay: a pure price-level shift (EV prices -20%) produced no
prediction-drift and no critical input-drift alert; only slice-level performance
caught it. Input drift
is necessary, not sufficient.

## `data_quality` — unseen categories, null-rate jumps

* Unseen category share above 2% (warn) / 10% (critical): a new country, make or
  lane is in production traffic that the model never trained on. Predictions for
  those rows fall back to "unknown" handling and are usually worse.
* Null rate up by 10 points: an online feature group stopped updating or a
  client stopped sending a field. Check the feature group's latest `event_time`
  in the online store and the materialisation job.

A critical data-quality incident that stays open for 5 weeks triggers a retrain,
because by then the new population is inside the next training window.

## `drift` / `prediction_drift` — PSI past calibrated thresholds

Thresholds are per feature, from a rolling backtest over the training period
(how much PSI moves within 8 weeks of a reference when nothing is wrong).
Market-context features (rolling medians) move a lot by design and have high
thresholds; vehicle attributes are stable and have low ones.

1. Look at which features moved and whether performance moved with them.
2. Drift without performance change: note it, no action. Drift with performance
   change: treat as `performance`.

## Endpoint alarms (CloudWatch, from Terraform)

* `*-5xx`: container errors. During a deployment the canary rolls back
  automatically. Otherwise check container logs; `aucml rollback <model> prod`
  restores the previous version in the registry, then re-run `promote`/apply.
* `*-p95-latency`: above the gate's latency budget. Check instance count against
  the autoscaling target and recent request batch sizes.

## Rollback

```bash
aucml rollback price_estimator prod --approved-by "<you>" --reason "<incident link>"
python scripts/render_tfvars.py prod --artifacts-bucket <bucket> > infra/terraform/environments/prod.models.tfvars.json
# then apply via the promote workflow
```
