# Final model decision

Model selection closed on November 5, 2026 after the planned E1-E4 experiments.

Retain the existing two-stage model: market-index adjustment and recent 13-week
price level, plus XGBoost predicting log rate per mile. Tree settings remain
depth 6, learning rate 0.03 and 1,500 trees. Features, cleaning and seed are
unchanged. Final fitting uses January-October labeled history after the
training-only outlier exclusion; the submission covers November-December.

## Evidence and decisions

| Check | Result | Decision |
|---|---|---|
| Three historical time folds | Mean all-row MAE $98.55, RMSE $623.62, MAPE 4.27% | Reference model |
| E1 withheld-city robustness | On the same 2,516 selected-city test rows, all-row MAPE rose from 4.75% to 4.99% after hiding those cities from training | Modest degradation on one fixed sample; no model change |
| E3 fixed 50:50 recent/trend blend | All-row mean MAPE improved to 4.20%, but filtered F1 worsened by 0.436 percentage points, above the 0.3-point limit | Reject under the preset protocol |
| E2 8/13/17-week windows | All-row mean MAPE 4.37% / 4.27% / 4.35% | Keep 13 weeks |
| E4 six tree settings | Depth 4, rate 0.03 slightly improved all-row mean MAPE to 4.2646%, but filtered mean improvement was only 0.0081 points, below the 0.05-point minimum | Keep original tree settings |

These are development backtests on overlapping time windows, repeatedly used
for model selection. They are not an untouched final test or the employer's
hidden-data score. Headline metrics include all evaluation rows. Filtered
metrics (reference mean MAPE 1.95%) use actual test prices to define a
retrospective diagnostic subset and must be labeled separately.

The acceptance protocol required at least 0.05 percentage points improvement
in filtered fold-mean MAPE, no filtered fold worsening above 0.3 points, and
no worsening of all-row fold-mean MAE or MAPE. This is a project decision rule,
not a statistical significance test. Small gains and rejected alternatives
should still be disclosed; the retained model is not best on every metric.

## Submission status

The selected model settings have not changed since the completed final fit.
Existing predictions therefore need no regeneration. The provided scorer
already passed for all 12,000 submission rows and 31 December chart rows;
this verifies file validity, not prediction accuracy. A preserved copy is in
`submission_v1/` in the working project. The final report is now included at
`output/pdf/Spotter_Freight_Rate_Assessment_Report.pdf`. Remaining external
deliverables are the accessible GitHub repository and Loom recording/link.

Evidence: `reports/reproduction.log`, `reports/city_holdout/`,
`reports/blend_experiment/`, `reports/window_experiment/`, and
`reports/tuning_experiment/`.
