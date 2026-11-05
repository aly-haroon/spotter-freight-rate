# Spotter freight-rate solution audit

Audit date: November 5, 2026. Evidence is the actual Python source, input CSVs,
saved logs and predictions, plus isolated diagnostic runs. No notebooks were
found outside the environment. Production source, data, saved results and
submission hashes were unchanged across the diagnostic run (71 protected files).
All new code and outputs are in `audit/`.

Added: `audit/run_audit.py` (isolated computation), `audit/build_report.py`
(report generator), `audit/AUDIT_REPORT.md`, CSV diagnostic tables and
predictions, JSON facts/protocol/hash manifest, scorer log and chart. Existing
model code, features, folds and saved prediction files were not changed.

Reproduce only these diagnostics from the project root:

```powershell
.\.venv\Scripts\python.exe -B -u -m audit.run_audit
.\.venv\Scripts\python.exe -B -m audit.build_report
```

MAE = average absolute dollar error; RMSE = square root of mean squared dollar
error, which gives large mistakes more influence; MAPE = average absolute
percentage error, calculated separately for each shipment; MedAE = median
absolute dollar error. ME = mean(predicted - actual): negative means underpricing.
All error tables include every evaluation row unless explicitly marked diagnostic.
Dollar metrics are rounded to cents; percentages are rounded, not estimated.

## 1. SOLUTION SUMMARY

The supplied `train-test.csv` and `validation.csv` are byte-identical to the
`data/train_test.csv` and `data/validation.csv` copies used by the code. Training
has 48,000 labeled rows, January 1-October 31, 2025; final data has 12,000 unlabeled
rows, November 1-December 31, 2025. IDs are unique in each, with no overlap.

**Final model:** `src/model.py::TwoStageModel`, with a market adjustment followed
by an XGBoost regressor. Training target is `log(posted_rate / distance)` minus
the fitted market offset. Prediction is `exp(tree prediction + market offset +
recent-level shift) * distance`. There is no log-to-mean bias correction.

**Tree features, complete list:** `log_distance`, `eq_Dry Van`, `eq_Flatbed`,
`eq_Reefer`, `weight`, `weight_missing`, `pickup_lat`, `pickup_lon`, `delivery_lat`,
`delivery_lon`, `delta_lat`, `delta_lon` (12 columns).

The separate market layer uses `log(market_index)`. Its coefficient is estimated
after five alternating demeaning sweeps for additive lane-equipment groups
(`pickup`, `delivery`, `equipment`) and calendar-week groups. This is not a
joint lane-by-week fixed effect. The recent-level shift is a mean over the last
13 weeks of lane-demeaned, market-adjusted training log rates. Date and city names
therefore affect this training calculation, but are not raw tree features.
No city/lane premium encoder, `quote_signal`, `load_id`, raw date, weekday, or
`t_months` is used as a final tree input. The straight-line trend option is off.

| Parameter | Final value |
| --- | --- |
| XGBoost loss | Default regressor squared error on transformed target |
| n_estimators | 1500 |
| learning_rate | 0.03 |
| max_depth | 6 |
| min_child_weight | 5 |
| subsample / colsample_bytree | 0.8 / 0.8 |
| tree_method | hist |
| random_state | 42 |
| n_jobs | -1 |
| Market level / recent weeks | recent / 13 |
| use_market_layer / use_premiums / extra_tree_features | True / False / [] |
| Other XGBoost parameters | Library defaults, XGBoost 3.4.1 |

The optional premium encoder uses five shuffled folds with seed 42, but is off
in the final model. City diagnostics and the audit classifier also use seed 42.

**Cleaning (`src/data.py::clean`):**

| Item | Rule | Training count | Final-data count |
| --- | --- | --- | --- |
| Negative weight | Take absolute value; assumed sign errors, not independently verified | 292 | 145 |
| Missing weight | Keep NaN for native XGBoost missing routing; add weight_missing | 300 | 165 |
| Missing market_index | Mean of observed market_index on the same date in the input batch | 374 | 249 |

`weight_was_negative` and `market_index_missing` are created but not final tree
features. No scaling is used. No observed market value or distance is nonpositive;
no date has all market values missing in either supplied dataset. That all-missing
date case has no fallback. Coordinates are not corrected geographically; each
city has one consistent coordinate pair across the supplied files.

Training-only exclusion: calculate rate per mile, then the median within equipment
x distance band x calendar month. Exclude rows with rate per mile outside
0.6-1.6 times that median. Distance-band edges are
`[0,150,250,350,500,700,1000,1300,1600,2000,2500,5000]`, right-closed.
Final training excludes 677 rows and fits on 47,323. Flagged prices are not proven
invalid. Test flags are used only to describe a target-conditioned diagnostic
subset and to evaluate experiment acceptance; all-row headline scores retain them.

**Splits:** expanding training windows, no omitted gap days. Each origin predicts
the following two calendar months without retraining within its test window.
This is a two-month forecast window, not a fixed two-month lead for every row.
Exact dates and counts appear in section 2. August and September overlap between
fold test windows; these are reused development folds, not an untouched final test.

**Simple baseline (`src/validate.py::BaselineModel`):** after the same training
outlier exclusion, store median rate per mile for each equipment x distance-band
cell. Predict that median times the shipment's distance. It uses no market,
location, weight or time adjustment and has no fallback for an absent cell.
No absent-cell failure occurred in the audited folds.

## 2. PER-FOLD RESULTS

| fold | model | Train dates (2025) | Test dates (2025) | Train raw -> fitted | n | MAE | RMSE | MAPE% | MedAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | baseline | 01-01 to 06-30 | 07-01 to 08-31 | 28806 -> 28406 | 9671 | 134.36 | 630.13 | 5.91 | 58.89 |
| F1 | model | 01-01 to 06-30 | 07-01 to 08-31 | 28806 -> 28406 | 9671 | 87.24 | 620.50 | 3.75 | 23.31 |
| F2 | baseline | 01-01 to 07-31 | 08-01 to 09-30 | 33718 -> 33253 | 9429 | 127.81 | 620.27 | 5.75 | 56.57 |
| F2 | model | 01-01 to 07-31 | 08-01 to 09-30 | 33718 -> 33253 | 9429 | 98.75 | 616.24 | 4.28 | 30.42 |
| F3 (MOST RECENT) | baseline | 01-01 to 08-31 | 09-01 to 10-31 | 38477 -> 37944 | 9523 | 133.55 | 638.21 | 6.03 | 57.95 |
| F3 (MOST RECENT) | model | 01-01 to 08-31 | 09-01 to 10-31 | 38477 -> 37944 | 9523 | 109.67 | 634.13 | 4.79 | 37.38 |

Reported headline = arithmetic mean of the three fold scores, each fold given equal weight. The RMSE below is the mean of fold RMSEs, not a pooled RMSE; the MedAE below is the mean of fold medians.

| model | MAE | RMSE | MAPE% | MedAE |
| --- | --- | --- | --- | --- |
| baseline | 131.91 | 629.54 | 5.90 | 57.80 |
| model | 98.55 | 623.62 | 4.27 | 30.37 |

For monthly and overall breakdowns, use each shipment's latest available forecast origin, chosen by date rather than error. There are 28,623 fold prediction occurrences, 19,194 unique shipments and 9,429 repeated occurrences. This retains F1 July, F2 August and F3 September-October. Thus the first three months use first-month forecasts; October uses second-month forecasts. The audit also reports pooling all occurrences, which double-counts some shipments.

| aggregation | model | n | MAE | RMSE | MAPE% | MedAE | ME |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all fold occurrences | baseline | 28623 | 131.93 | 629.61 | 5.90 | 57.71 | -39.52 |
| all fold occurrences | model | 28623 | 98.49 | 623.68 | 4.27 | 29.79 | -57.32 |
| unique latest-origin | baseline | 19194 | 134.44 | 634.13 | 5.99 | 58.45 | -46.86 |
| unique latest-origin | model | 19194 | 98.40 | 627.18 | 4.27 | 29.30 | -56.72 |

## 3. ERROR OVER TIME

Unique latest-origin forecasts; ME is signed dollars.

| month | model | n | MAE | MAPE% | ME |
| --- | --- | --- | --- | --- | --- |
| 2025-07 | baseline | 4912 | 144.03 | 5.97 | -88.79 |
| 2025-07 | model | 4912 | 91.52 | 3.65 | -49.32 |
| 2025-08 | baseline | 4759 | 126.31 | 5.95 | 2.55 |
| 2025-08 | model | 4759 | 82.93 | 3.85 | -19.29 |
| 2025-09 | baseline | 4670 | 129.75 | 5.56 | -46.88 |
| 2025-09 | model | 4670 | 112.33 | 4.60 | -85.26 |
| 2025-10 | baseline | 4853 | 137.21 | 6.48 | -52.86 |
| 2025-10 | model | 4853 | 107.11 | 4.97 | -73.43 |

Percentage error rises later in the period; dollar MAE is not monotonically increasing. Both recent months show systematic underprediction. The effect is not solely extreme prices: F3's non-flagged diagnostic subset has mean signed error -$45.61 (versus -$79.23 across all F3 rows). These observations do not establish what November-December accuracy will be.

## 4. ERROR BREAKDOWNS

All rows included. Price buckets use actual posted_rate (retrospective diagnosis
only); distance is miles and weight is pounds after absolute-value cleaning.
Missing weights are separate. A route is the ordered pickup-delivery pair,
regardless of equipment; 'seen' means present in raw training rows before outlier
exclusion, matching the original backtest definition. Small buckets are unstable.

### Most recent fold: September-October

| dimension | bucket | n | model_MAE | model_MAPE% | baseline_MAE | baseline_MAPE% |
| --- | --- | --- | --- | --- | --- | --- |
| equipment | Dry Van | 5360 | 106.23 | 4.45 | 132.07 | 5.81 |
| equipment | Flatbed | 1770 | 110.28 | 5.38 | 127.16 | 6.30 |
| equipment | Reefer | 2393 | 116.92 | 5.09 | 141.58 | 6.30 |
| distance_bucket | <=250 | 550 | 21.73 | 4.50 | 29.09 | 6.11 |
| distance_bucket | (250,500] | 1540 | 53.40 | 5.52 | 66.92 | 7.03 |
| distance_bucket | (500,1000] | 2972 | 74.04 | 4.91 | 97.17 | 6.42 |
| distance_bucket | (1000,2000] | 2927 | 148.51 | 4.57 | 173.08 | 5.47 |
| distance_bucket | >2000 | 1534 | 192.61 | 4.34 | 232.94 | 5.28 |
| price_bucket | <1000 | 1545 | 67.79 | 13.28 | 79.45 | 15.07 |
| price_bucket | [1000,2000) | 3077 | 43.28 | 2.95 | 67.09 | 4.55 |
| price_bucket | [2000,4000) | 3387 | 72.77 | 2.55 | 97.54 | 3.43 |
| price_bucket | >=4000 | 1514 | 369.90 | 4.86 | 404.39 | 5.60 |
| weight_bucket | (10000,25000] | 1950 | 101.11 | 5.37 | 132.64 | 7.04 |
| weight_bucket | (25000,40000] | 5980 | 111.22 | 4.59 | 127.89 | 5.54 |
| weight_bucket | <=10000 | 37 | 132.26 | 12.06 | 182.00 | 15.40 |
| weight_bucket | >40000 | 1491 | 115.29 | 4.70 | 158.51 | 6.52 |
| weight_bucket | missing | 65 | 82.25 | 3.19 | 81.40 | 3.36 |
| weight_quality | originally missing | 65 | 82.25 | 3.19 | 81.40 | 3.36 |
| weight_quality | originally negative | 59 | 103.72 | 9.57 | 124.33 | 10.75 |
| weight_quality | originally nonnegative | 9399 | 109.90 | 4.77 | 133.97 | 6.01 |
| route_status | seen | 9502 | 109.80 | 4.79 | 133.69 | 6.03 |
| route_status | unseen | 21 | 50.21 | 2.11 | 69.22 | 2.94 |

### Overall unique out-of-fold

| dimension | bucket | n | model_MAE | model_MAPE% | baseline_MAE | baseline_MAPE% |
| --- | --- | --- | --- | --- | --- | --- |
| equipment | Dry Van | 10862 | 97.89 | 4.08 | 134.38 | 5.89 |
| equipment | Flatbed | 3512 | 90.96 | 4.76 | 122.95 | 6.28 |
| equipment | Reefer | 4820 | 104.96 | 4.32 | 142.94 | 6.01 |
| distance_bucket | <=250 | 1083 | 19.02 | 4.02 | 29.45 | 6.22 |
| distance_bucket | (250,500] | 3089 | 40.82 | 4.69 | 59.28 | 6.71 |
| distance_bucket | (500,1000] | 5978 | 65.17 | 4.42 | 97.16 | 6.41 |
| distance_bucket | (1000,2000] | 6022 | 142.78 | 4.04 | 183.62 | 5.45 |
| distance_bucket | >2000 | 3022 | 162.99 | 4.06 | 224.60 | 5.42 |
| price_bucket | <1000 | 3117 | 59.67 | 11.95 | 74.07 | 14.07 |
| price_bucket | [1000,2000) | 6227 | 39.03 | 2.70 | 70.36 | 4.80 |
| price_bucket | [2000,4000) | 6915 | 60.75 | 2.11 | 101.20 | 3.52 |
| price_bucket | >=4000 | 2935 | 354.16 | 4.50 | 412.80 | 5.74 |
| weight_bucket | (10000,25000] | 3995 | 92.25 | 4.62 | 135.48 | 6.82 |
| weight_bucket | (25000,40000] | 12117 | 97.84 | 4.17 | 127.24 | 5.61 |
| weight_bucket | <=10000 | 82 | 81.23 | 6.25 | 160.96 | 10.22 |
| weight_bucket | >40000 | 2877 | 109.92 | 4.11 | 163.89 | 6.35 |
| weight_bucket | missing | 123 | 95.10 | 4.46 | 102.67 | 4.90 |
| weight_quality | originally missing | 123 | 95.10 | 4.46 | 102.67 | 4.90 |
| weight_quality | originally negative | 96 | 73.41 | 6.33 | 104.75 | 7.73 |
| weight_quality | originally nonnegative | 18975 | 98.54 | 4.25 | 134.79 | 5.99 |
| route_status | seen | 19126 | 98.60 | 4.27 | 134.65 | 6.00 |
| route_status | unseen | 68 | 40.77 | 1.71 | 73.77 | 3.15 |

## 5. UNSEEN CITIES AND ROUTES

Route errors are in section 4; F3 has only 21 unseen-route examples and overall unique OOF has 68. Their lower error does not prove unseen routes are easier. Final data has 1,447 rows touching a new city and 1,461 on unseen ordered routes. New cities: Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego.

One OOF shipment's route appears in raw training but disappears after outlier filtering. If 'seen' instead requires a row in the actual fitted training data, model results are:

| scope | route status after filtering | n | MAE | MAPE% |
| --- | --- | --- | --- | --- |
| F3 | seen | 9501 | 109.81 | 4.79 |
| F3 | unseen | 22 | 51.07 | 2.14 |
| unique OOF | seen | 19125 | 98.60 | 4.27 |
| unique OOF | unseen | 69 | 41.18 | 1.73 |

**New three-city diagnostic:** hide Bakersfield, Montgomery, Providence from both pickup and delivery in January-August, then score the same 999 September-October rows touching those cities. Cities were sampled before fitting, from sorted historical city names with seed 42. Restricted training has 34,581 raw rows and 34,091 after outlier exclusion.

| model | training_rows | n | MAE | RMSE | MAPE% | MedAE | ME |
| --- | --- | --- | --- | --- | --- | --- | --- |
| reference | 37944 | 999 | 167.02 | 1,026.14 | 4.15 | 46.27 | -134.53 |
| 3 cities withheld | 34091 | 999 | 169.05 | 1,028.37 | 4.17 | 46.77 | -133.51 |

Existing eight-city diagnostic is also retained as separate evidence. Hidden cities: Bakersfield, Hartford, Memphis, New York, San Antonio, Toledo, Tulsa, Washington. On the same selected-city test rows:

| model | training_rows | n | MAE | RMSE | MAPE% |
| --- | --- | --- | --- | --- | --- |
| reference | 37944 | 2516 | 114.17 | 665.80 | 4.75 |
| cities_hidden | 28018 | 2516 | 119.61 | 667.01 | 4.99 |

Both tests show a modest error increase. Removing cities also removes training rows; these single city selections do not isolate geography from sample size and do not guarantee accuracy on the eight actual final-only cities.

## 6. LEAKAGE AND REPRODUCIBILITY AUDIT

| Question | Verdict | Evidence / limitation |
| --- | --- | --- |
| Are all imputations fit only on fold training rows? | NO | `clean` computes same-date market means from the frame being cleaned. `src.validate.main` cleans the whole historical frame before splitting; dates do not cross fold boundaries, so no future date enters a training-date mean. Test-date means still use test-batch features. This is batch-dependent preprocessing, not target leakage. |
| Are active fitted model/encoding statistics training-only? | YES | Training outlier references, market slope, recent level and trees use the fold's training rows. Final premiums are off; optional premium tables are fold-training-only and OOF inside that training data. No scaler or fitted weight imputer exists. |
| Are date-trend, load_id or leakage-prone inputs used? | YES, with qualification | Date is used for historical weeks/recent level and same-date market imputation; final linear trend is off. `t_months` is computed but unused by the selected model. `load_id` is only an output/join key. `quote_signal` is excluded. Availability/provenance of future market_index is not independently verified. |
| Are any validation.csv labels used? | NO | No posted_rate column exists. Only final feature values are used, including date-level market means and city coordinates for December. |
| Are test labels used anywhere? | YES | Test prices are used for scoring, outlier diagnostic flags and selection among development variants. They are not model prediction inputs. Filtered scores therefore must not be presented as all-load performance. |
| Can final fitting reproduce saved predictions with fixed seeds? | YES on this environment | Two fresh final fits had maximum unrounded prediction difference $0.00. Refit predictions rounded to cents matched every saved final and December prediction exactly. F3 refit maximum difference from saved CSV floats was 9.09e-13 dollars. |
| Is a completely fresh-machine rebuild verified? | UNCLEAR | Audit reused the installed environment. A lock file exists, but a new environment, hardware and OS were not tested. |

Runtime: Python 3.12.14, NumPy 2.5.4,
pandas 2.3.3, scikit-learn 1.9.1,
XGBoost 3.4.1, SciPy 1.18.1.
No production training command was invoked because it would overwrite submissions;
the audit imported and fitted its unchanged classes and reconstructed its output
mapping and rounding in memory. No serialized fitted model was found among the
project files; reproducibility relies on the code, inputs and environment.

## 7. DISTRIBUTION SHIFT

Adversarial validation means predicting whether a row comes from labeled history
or final data. AUC near 0.5 means little separability; a higher value means the
feature distributions are distinguishable, not necessarily that regression fails.

Audit classifier: XGBoost binary classifier, 300 trees, depth 3, learning rate
0.05, subsample and colsample 0.8, hist, seed 42, four threads. A single
GroupShuffleSplit holds out 30% of complete dates, seed 42; no date is in both
classifier partitions. Raw date, load ID, target and target-derived fields are
excluded. 'Deployed features' means the 12 tree inputs plus log(market_index).
The second variant adds the excluded quote_signal to diagnose its shift.

| features | AUC | classifier_train_n | classifier_test_n | train_source_in_classifier_test | final_source_in_classifier_test | training_dates | test_dates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deployed features | 0.7738 | 41571 | 18429 | 13676 | 4753 | 255 | 110 |
| deployed plus unused quote_signal | 0.8194 | 41571 | 18429 | 13676 | 4753 | 255 | 110 |

Top features by held-out permutation importance, averaged over three shuffles. AUC drop measures how much separability is lost after shuffling that feature; correlated features can share importance.

| features | feature | heldout_permutation_AUC_drop |
| --- | --- | --- |
| deployed features | log_market_index | 0.2478 |
| deployed features | delivery_lat | 0.0084 |
| deployed features | pickup_lat | 0.0052 |
| deployed plus unused quote_signal | log_market_index | 0.2477 |
| deployed plus unused quote_signal | quote_signal | 0.0545 |
| deployed plus unused quote_signal | log_distance | 0.0388 |

Raw observed distributions below exclude missing values. SD is sample standard deviation; KS is the largest distribution-function difference relative to September-October (not a p-value).

| feature | period | n_observed | n_missing | mean | std | p05 | median | p95 | KS_vs_SepOct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market_index | Sep-Oct | 9450 | 73 | 0.9259 | 0.0822 | 0.7896 | 0.9281 | 1.0528 | 0.0000 |
| market_index | Nov | 5721 | 115 | 0.9187 | 0.0739 | 0.8083 | 0.9123 | 1.0336 | 0.0622 |
| market_index | Dec | 6030 | 134 | 0.9347 | 0.0735 | 0.8215 | 0.9315 | 1.0475 | 0.0695 |
| market_index | Nov-Dec | 11751 | 249 | 0.9269 | 0.0741 | 0.8145 | 0.9229 | 1.0409 | 0.0527 |
| quote_signal | Sep-Oct | 9523 | 0 | 2.0688 | 0.3031 | 1.5777 | 2.0653 | 2.5745 | 0.0000 |
| quote_signal | Nov | 5836 | 0 | 2.0553 | 0.2203 | 1.6947 | 2.0575 | 2.4208 | 0.0488 |
| quote_signal | Dec | 6164 | 0 | 2.0476 | 0.2193 | 1.6856 | 2.0468 | 2.4081 | 0.0581 |
| quote_signal | Nov-Dec | 12000 | 0 | 2.0513 | 0.2198 | 1.6917 | 2.0512 | 2.4137 | 0.0526 |

The strong adversarial result compares all January-October history against November-December. Marginal market distributions are closer when comparing only September-October against the final period. Neither check measures unobserved final target relationships. Verified quote_signal x distance proxy:

| month | n | MAE | MAPE% | quote_rpm_correlation |
| --- | --- | --- | --- | --- |
| 2025-02 | 4337 | 55.2426 | 2.8481 | 0.4790 |
| 2025-10 | 4853 | 413.7224 | 20.9937 | -0.4340 |

The historical quote-price association changes sign. With no final labels, its final relationship to actual price is NOT AVAILABLE; a feature-only proxy cannot prove the signal is independent of the final target.

## 8. DECEMBER PIPELINE

`src/predict.py::build_december_features` fills the chart as follows:

| Field | Exact source |
| --- | --- |
| pickup, delivery, distance, equipment, weight, date | Original chart inputs; Lexington -> Fort Wayne, 360 miles, Dry Van, 32,000 pounds; one row per December date |
| Lexington coordinates | (36.99152, -84.99876), city lookup from concatenated filtered historical training and unlabeled final rows, first occurrence per city |
| Fort Wayne coordinates | (41.31561, -85.36206), same lookup rule |
| market_index | Date's mean from cleaned validation.csv feature rows. Cleaning first fills missing market values with that same date's observed mean, so this is also the raw observed daily mean. |
| quote_signal | NaN; excluded from final model, so it has no prediction effect |

`audit/december_features.csv` records all 31 completed feature rows;
`audit/december_market_sources.csv` records every source date's number of loads,
number with an observed market value, and mean. All city-coordinate pairs are
consistent across the supplied files. The chart uses contemporaneous final-batch
market information: it is conditional on those supplied features, not a historical-
only forecast of December's market index.

**Latest-holdout simulation:** fit the unchanged model on January-August. Predict
September-October using normal row-level market values, then replace each with
its holdout-date mean and set quote_signal to NaN, exactly as the December rule.
Coordinates remain the consistent city coordinates. This tests information loss
from the chart approximation, not forecasting unknown future market values.

| scope | features | n | MAE | MAPE% |
| --- | --- | --- | --- | --- |
| all F3 | true row features | 9523 | 109.6697 | 4.7870 |
| all F3 | December-like daily market mean; quote missing | 9523 | 110.0382 | 4.8003 |
| Lexington to Fort Wayne F3 | true row features | 6 | 28.6552 | 3.3050 |
| Lexington to Fort Wayne F3 | December-like daily market mean; quote missing | 6 | 27.9670 | 3.2172 |

Positive change means degradation; negative means improvement.

| scope | MAE change $ | MAPE change percentage points |
| --- | --- | --- |
| all F3 | 0.3685 | 0.0132 |
| Lexington to Fort Wayne F3 | -0.6881 | -0.0878 |

Historical Lexington -> Fort Wayne comparison (rate per mile = total dollars / actual miles; rows differ in equipment, distance and weight):

| scope | n | price_min | price_median | price_max | rpm_min | rpm_median | rpm_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all historical route rows | 32 | 757.9300 | 848.6100 | 973.7500 | 2.1154 | 2.3241 | 2.6678 |
| historical route Dry Van | 21 | 757.9300 | 807.8900 | 934.3700 | 2.1154 | 2.2334 | 2.4667 |
| December fixed chart | 31 | 811.3700 | 821.7700 | 833.8700 | 2.2538 | 2.2827 | 2.3163 |

Only 21 of the 32 historical route rows are Dry Van. Their distances span 350.9-378.8 miles and weights 16,195-41,021 pounds, so they are not identical to the fixed chart shipment. Historical rows are in `audit/historical_chart_route.csv`.

**Day-to-day shape:** a repeating weekly wave, with low Sunday/Monday and
high Wednesday/Thursday values, rather than abrupt isolated spikes. All tree
inputs are fixed, so all day-to-day movement comes from the daily market mean
through the market layer; weekday and holidays are not model features. The final
market exponent is 0.119355. Adjacent-day changes range
from $-9.06 to +$8.89; median absolute
daily change is $6.585. December 1 is
$811.87 and December 31 is $832.21; the fitted linear slope
is only $0.0974/day, so endpoint comparison
alone overstates the overall trend. No holiday-specific response is learned.

## 9. OUTPUT FILE CHECKS

| Check | Result |
| --- | --- |
| validation_predictions.csv rows | PASS: 12,000 |
| Exact columns and order | PASS: load_id,predicted_rate |
| IDs TE-000001 through TE-012000 once each | PASS: exact set; no duplicates or missing IDs |
| All predictions finite and greater than zero | PASS |
| December fixed rows, dates and seven-column structure | PASS: supplied score.py, exit code 0 |
| Prediction reproduction | PASS: exact saved values after cent rounding |
| Production files changed by audit | NO: all 71 protected file hashes unchanged |

Prediction distributions (total dollars):

| file | n | min | median | max |
| --- | --- | --- | --- | --- |
| validation_predictions.csv | 12000 | 185.220 | 2,024.785 | 6,838.210 |
| data/december_chart_inputs.csv | 31 | 811.370 | 821.770 | 833.870 |

Scorer rerun log: `audit/scorer.log`; its new chart is `audit/scorer_results/candidate_december.png`. The scorer checks file validity and draws the chart; it does not reveal hidden-label accuracy.

## 10. TOP RISKS AND IMPROVEMENT IDEAS

These are ranked audit judgments, not additional model changes. Future improvement
estimates are NOT AVAILABLE without testing a specific intervention on new data.
The measured comparisons below are evidence, not promised gains.

| Rank | Weakness / evidence | Improvement idea / measured scale |
| --- | --- | --- |
| 1 | Time drift and underprediction: section 3 shows higher recent percentage errors and negative signed errors, including the non-flagged F3 subset. | Validate a separately specified level-calibration method on new future origins. The tested blend lowered all-row mean MAE by $2.28 and MAPE by 0.0670 points, but failed the preset F1 stability rule. Future gain: NOT AVAILABLE. |
| 2 | Extreme-price tail: 277 of 19194 unique OOF rows were flagged, contributing 55.54% of total absolute error; flagged-row model MAE was $3,787.02. | Investigate price provenance and omitted pricing conditions; distinguish legitimate extremes from errors. Do not delete these rows from headline evaluation. Validated tail remedy and future gain estimate: NOT AVAILABLE. |
| 3 | Future-feature availability and shift: December borrows same-date final-batch market information; adversarial separation is driven primarily by market_index. | Document the batch-conditional contract; separately evaluate a historical-only market forecast if needed. The within-holdout daily-mean approximation changed MAE by +$0.3685 and MAPE by +0.01321 points. Gain for a real advance market forecast: NOT AVAILABLE. |
| 4 | Limited independent validation: overlapping development folds were reused for variant selection, and acceptance relied partly on a label-filtered diagnostic score. No untouched final labeled set exists. | Use new chronological origins or external labeled data for an independent evaluation; avoid claiming filtered error as full-population accuracy. Expected score correction or future improvement: NOT AVAILABLE. |
| 5 | Sparse geography and chart evidence: very few naturally unseen historical routes, one three-city and one eight-city selection, and only six F3 rows on the chart route. | Repeat fixed city holdouts with size-matched controls before changing geographic modeling. Three-city withholding increased selected-row MAE by $2.03 and MAPE by 0.02189 points; the eight-city increase was $5.44 and 0.23586 points. Future gain: NOT AVAILABLE. |

**Diminishing returns from tree tuning:** the best depth-4 candidate changed all-row fold-mean MAE from $98.5519 to $98.3187 (a $0.2332 reduction) and MAPE from 4.271289% to 4.264583% (0.006706 percentage points). Its filtered improvement was 0.008108 points, below the preset 0.05-point minimum. It was faster and slightly better, so retaining depth 6 is a protocol decision, not proof that depth 6 has the lowest error. Both alternative recent windows were worse overall. Further blind tree/window tuning on the same folds has little supporting evidence; resolving drift, tail provenance and validation assumptions matters more. Keep the current submissions while reviewing these risks.

## UNKNOWNS

- **NOT AVAILABLE:** final November-December MAE/MAPE, employer's hidden metric,
  benchmark, business error tolerance or hiring threshold; no final labels supplied.
- **NOT AVAILABLE:** proof of the negative-weight sign-error assumption or that
  flagged prices are invalid; no source-system documentation or investigation.
- **NOT AVAILABLE:** actual final quote_signal-to-price relationship; final prices
  do not exist in the workspace. Historical drift and feature-only checks do not
  establish final target independence or the signal's generation provenance.
- **NOT AVAILABLE:** independent confirmation that each November-December market
  value would be known at the original forecast origin. The batch is supplied,
  but a historical-only market forecast was not evaluated.
- **NOT AVAILABLE:** an untouched labeled final test, fresh-machine reproduction,
  formal confidence intervals, a business cost function, or defensible numeric
  estimates of gains from unimplemented improvements.
- **NOT AVAILABLE:** generalization across all possible held-out-city choices,
  holidays, future years or changing geographical relationships; current evidence
  is limited to the supplied dates and selected diagnostics.
