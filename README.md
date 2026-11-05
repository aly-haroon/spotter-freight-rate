# Freight Rate Prediction

Spotter ML Engineer assessment: predict total `posted_rate` for November-December 2025 shipments using January-October labeled history and supplied load features.

The selected solution combines a market adjustment and a recent 13-week price level with XGBoost. Completed predictions, the December chart and the final PDF report are included. No training is needed to inspect or validate the submission.

## Submission files

| Deliverable | Location |
| --- | --- |
| Prices for all 12,000 final shipments | [validation_predictions.csv](validation_predictions.csv) |
| Completed 31-row December inputs | [data/december_chart_inputs.csv](data/december_chart_inputs.csv) |
| Chart produced by the supplied scorer | [scorer_results/candidate_december.png](scorer_results/candidate_december.png) |
| Final assessment report | [PDF report](output/pdf/Spotter_Freight_Rate_Assessment_Report.pdf) |
| Original assessment instructions | [Assessment brief](docs/assessment_instructions.pdf) |

The camera-on Loom recording is a separate required deliverable. Its link will be supplied with the final submission after recording.

## Historical results

Arithmetic means of three chronological development-fold scores; every test row is included, including flagged extreme prices.

| Model | MAE ($) | RMSE ($) | MAPE |
| --- | ---: | ---: | ---: |
| Equipment-distance median baseline | 131.91 | 629.54 | 5.90% |
| Selected two-stage model | **98.55** | **623.62** | **4.27%** |

MAE is average absolute dollar error; MAPE averages each shipment's absolute percentage error. RMSE gives large mistakes more influence; the table averages fold RMSEs, rather than pooling squared errors. Final November-December accuracy is unknown because no final prices are supplied.

## Setup

Run commands from the repository root. Python **3.12.14** was used for the verified run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`. If PowerShell activation is unavailable, use `.\.venv\Scripts\python.exe` instead of `python` without activating.

`requirements-lock.txt` records the exact installed versions used for the saved results. `requirements.txt` gives broader dependency ranges. Other versions or hardware have not been verified. No GPU, API key or external service is required. Inputs are included under `data/`.

## Check the completed submission

No training or prediction overwriting:

```powershell
python -B -m audit.verify_submission
```

This checks input schemas and dates, output IDs and positivity, release file hashes, and Python syntax. It calls the supplied scorer's validation functions and writes its check result under `audit/submission_checks/`.

To run the original scorer and regenerate its chart:

```powershell
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates structure and generates a figure. It does **not** calculate hidden-label accuracy.

## Reproduce analysis, validation and predictions

```powershell
python -m src.eda
python -u -m src.validate
python -m src.predict
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

EDA writes exploratory tables under `reports/`. Validation trains and evaluates nine predefined variants. Prediction generation fits the selected model and **overwrites** `validation_predictions.csv` and `data/december_chart_inputs.csv`; the scorer overwrites its chart. Fitted models are not persisted, so prediction generation retrains the model.

## Validation design

| Fold | Training dates, 2025 | Test dates, 2025 | Training rows raw / fitted | Test rows |
| --- | --- | --- | ---: | ---: |
| F1 | January 1-June 30 | July 1-August 31 | 28,806 / 28,406 | 9,671 |
| F2 | January 1-July 31 | August 1-September 30 | 33,718 / 33,253 | 9,429 |
| F3 | January 1-August 31 | September 1-October 31 | 38,477 / 37,944 | 9,523 |

There is no gap. Each model predicts the next two calendar months without retraining within its test window. Windows overlap and were reused for development; they are not independent untouched tests. Monthly audits keep the latest available forecast per shipment to avoid duplicate rows.

## Model and data decisions

- **Target:** `log(posted_rate / distance)`; convert back to total dollars. No log-to-mean bias correction is applied.
- **Market layer:** estimate the `log(market_index)` association after demeaning additive lane-equipment and calendar-week effects; apply a recent 13-week price-level shift.
- **Tree inputs:** log distance; equipment indicators; weight and its missing indicator; pickup/delivery coordinates; coordinate differences. Raw dates, IDs, quote signals and target encodings are not final tree inputs.
- **XGBoost:** 1,500 trees, learning rate 0.03, depth 6, minimum child weight 5, row/column subsampling 0.8, histogram trees, seed 42, all available CPU threads.
- **Weight:** take absolute values for negatives, assuming sign errors; retain missing values for native tree handling.
- **Missing market index:** use the same date's mean observed feature value in the input batch. Test imputation uses test-batch features, not labels. Entirely missing dates have no fallback and were not present in the supplied data.
- **Flagged prices:** exclude training RPM outside 0.6-1.6 times its equipment-distance-band-month median. Final training retains 47,323 of 48,000 loads. Flagged prices are not proven invalid; headline evaluation includes them.
- **quote_signal:** exclude because its historical price relationship changes by month. Exploratory regime labels are heuristics; there are no final labels to establish its November-December target relationship.

## December chart assumptions

The fixed shipment is Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 pounds. Coordinates come from consistent city observations in the supplied history and final feature batch. Each December date uses the same-date mean market index from final feature rows. `quote_signal` is NaN and unused.

The chart conditions on supplied future market information; it is not a historical-only forecast of an unknown December market index. Its weekly variation comes from those market values, not an explicit weekday or holiday feature. The PDF includes a holdout simulation and historical-route comparison.

## Experiments and diagnostic evidence

| Command | Purpose |
| --- | --- |
| `python -u -m src.city_holdout` | Fixed eight-city exclusion on September-October |
| `python -u -m src.blend_experiment` | Fixed recent/trend blend |
| `python -u -m src.window_experiment` | Compare 8-, 13- and 17-week price windows |
| `python -u -m src.tuning_experiment` | Fixed six-setting XGBoost comparison |

These commands leave submission predictions unchanged. `--prepare-only` checks their splits without fitting. Saved protocols and metrics are under `reports/`. Detailed audits are in [audit/AUDIT_REPORT.md](audit/AUDIT_REPORT.md) and [audit/residual_tail/REPORT.md](audit/residual_tail/REPORT.md).

To recompute the audit, retain the supplied tuning prediction files (or rerun `src.tuning_experiment`), then run:

```powershell
python -B -u -m audit.run_audit
python -B -m audit.build_report
python -B -u -m audit.residual_tail_audit
```

These commands include diagnostic model fits and write only under `audit/`. They do not change the final model or submission predictions. Reading the prebuilt PDF requires no document-authoring dependencies.

## Limitations and reproducibility

Recent-month predictions tend to underprice, extreme prices remain difficult, and final data includes cities absent from history. City holdouts offer limited robustness evidence. Filtered diagnostic MAPE of about 1.95% must not replace the all-row headline of 4.27%. Small tuning gains and the rejected blend are disclosed in the PDF and [model decision record](docs/MODEL_DECISION.md).

Two fresh final fits reproduced the saved predictions in the verified environment. A fresh-machine or cross-hardware rebuild has not been tested. The release manifest records packaged file hashes; intentionally regenerating tracked outputs may change those hashes.

## Repository layout

```text
data/                         Inputs; december_chart_inputs.csv is completed
src/                          Cleaning, features, model and experiments
reports/                      EDA, backtest and experiment evidence
audit/                        Diagnostic code, tables and submission verifier
docs/                         Assessment instructions and model decision
output/pdf/                   Final assessment PDF
scorer_results/               December chart from the supplied scorer
validation_predictions.csv    Completed 12,000-row predictions
score.py                      Original supplied validator
requirements.txt              Dependency ranges
requirements-lock.txt         Verified installed versions
submission_manifest.json      SHA-256 release file hashes
```
