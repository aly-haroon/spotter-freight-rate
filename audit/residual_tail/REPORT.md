# Residual and tail diagnostics

## 1. Weekday and holiday residuals

Latest-origin OOF; all rows; signed % = 100 × mean((predicted − actual)/actual).

| weekday | n | mean_signed_pct_error | MAPE_pct |
| --- | --- | --- | --- |
| Monday | 2583 | 0.3938 | 4.4648 |
| Tuesday | 2736 | 0.0623 | 4.1529 |
| Wednesday | 2867 | 0.0047 | 4.1688 |
| Thursday | 2877 | 0.6057 | 4.5089 |
| Friday | 2783 | 0.2489 | 3.7460 |
| Saturday | 2697 | 0.9010 | 4.4153 |
| Sunday | 2651 | 0.2114 | 4.4189 |

Holiday windows: both endpoints included.

| holiday | holiday_date | start | end | n | mean_signed_pct_error | MAPE_pct |
| --- | --- | --- | --- | --- | --- | --- |
| Memorial Day | 2025-05-26 | 2025-05-23 | 2025-05-29 | 0 | NOT AVAILABLE | NOT AVAILABLE |
| July 4 | 2025-07-04 | 2025-07-01 | 2025-07-07 | 1114 | 0.8482 | 4.2067 |
| Labor Day | 2025-09-01 | 2025-08-29 | 2025-09-04 | 1052 | 0.8190 | 3.0286 |

| holiday | offset_days | date | n | mean_signed_pct_error | MAPE_pct |
| --- | --- | --- | --- | --- | --- |
| Memorial Day | -3 | 2025-05-23 | 0 | NOT AVAILABLE | NOT AVAILABLE |
| Memorial Day | -2 | 2025-05-24 | 0 | NOT AVAILABLE | NOT AVAILABLE |
| Memorial Day | -1 | 2025-05-25 | 0 | NOT AVAILABLE | NOT AVAILABLE |
| Memorial Day | 0 | 2025-05-26 | 0 | NOT AVAILABLE | NOT AVAILABLE |
| Memorial Day | 1 | 2025-05-27 | 0 | NOT AVAILABLE | NOT AVAILABLE |
| Memorial Day | 2 | 2025-05-28 | 0 | NOT AVAILABLE | NOT AVAILABLE |
| Memorial Day | 3 | 2025-05-29 | 0 | NOT AVAILABLE | NOT AVAILABLE |
| July 4 | -3 | 2025-07-01 | 135 | -1.6010 | 2.6583 |
| July 4 | -2 | 2025-07-02 | 171 | 3.9185 | 7.0542 |
| July 4 | -1 | 2025-07-03 | 172 | 0.4817 | 4.4873 |
| July 4 | 0 | 2025-07-04 | 160 | -1.1413 | 1.9878 |
| July 4 | 1 | 2025-07-05 | 157 | 4.3604 | 6.1774 |
| July 4 | 2 | 2025-07-06 | 158 | -1.3919 | 2.3522 |
| July 4 | 3 | 2025-07-07 | 161 | 0.7832 | 4.2842 |
| Labor Day | -3 | 2025-08-29 | 151 | -0.6346 | 1.8047 |
| Labor Day | -2 | 2025-08-30 | 140 | 5.1280 | 6.5954 |
| Labor Day | -1 | 2025-08-31 | 147 | -0.8328 | 1.8098 |
| Labor Day | 0 | 2025-09-01 | 157 | 2.3422 | 4.0822 |
| Labor Day | 1 | 2025-09-02 | 153 | 0.1675 | 2.9353 |
| Labor Day | 2 | 2025-09-03 | 148 | 0.2934 | 2.8331 |
| Labor Day | 3 | 2025-09-04 | 156 | -0.4801 | 1.3774 |

Memorial Day: NOT AVAILABLE; OOF begins 2025-07-01.

Market-adjusted RPM = posted_rate / distance / market_index**0.119354786573; training 2025-01-01–2025-10-31.

| population | weekday | n | mean_rpm | mean_market_adjusted_rpm | median_market_adjusted_rpm | adjusted_mean_vs_population_mean_pct |
| --- | --- | --- | --- | --- | --- | --- |
| all training rows | Monday | 6769 | 2.1964 | 2.1991 | 2.1294 | 0.1009 |
| all training rows | Tuesday | 6737 | 2.2134 | 2.1964 | 2.1203 | -0.0226 |
| all training rows | Wednesday | 7007 | 2.2484 | 2.2124 | 2.1290 | 0.7062 |
| all training rows | Thursday | 6934 | 2.2446 | 2.2028 | 2.1284 | 0.2664 |
| all training rows | Friday | 6975 | 2.2149 | 2.1837 | 2.1258 | -0.6002 |
| all training rows | Saturday | 6770 | 2.1957 | 2.1858 | 2.1179 | -0.5085 |
| all training rows | Sunday | 6808 | 2.1923 | 2.1979 | 2.1321 | 0.0443 |
| non-flagged training rows | Monday | 6676 | 2.1710 | 2.1737 | 2.1295 | 0.2392 |
| non-flagged training rows | Tuesday | 6637 | 2.1833 | 2.1665 | 2.1200 | -0.0893 |
| non-flagged training rows | Wednesday | 6901 | 2.2009 | 2.1656 | 2.1283 | -0.1310 |
| non-flagged training rows | Thursday | 6824 | 2.2102 | 2.1690 | 2.1284 | 0.0258 |
| non-flagged training rows | Friday | 6883 | 2.1990 | 2.1680 | 2.1267 | -0.0209 |
| non-flagged training rows | Saturday | 6687 | 2.1750 | 2.1650 | 2.1182 | -0.1584 |
| non-flagged training rows | Sunday | 6715 | 2.1658 | 2.1715 | 2.1321 | 0.1379 |

## 2. Tail predictability

14 classifier inputs: 12 tree features + log(market_index) + quote_signal; seed 42; no gap; no flagged training rows excluded.

| train_start | train_end | test_start | test_end | train_n | train_flagged_n | test_n | test_flagged_n | test_prevalence_pct | AUC | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 38477 | 533 | 9523 | 144 | 1.512129 | 0.501462 | 0.016244 |

Typical price = equipment × distance-band × month median RPM × row distance; shares use flagged rows as denominator.

| population | rows | flagged_n | above_typical_n | below_typical_n | above_share_of_flagged_pct | below_share_of_flagged_pct | unclassified_flagged_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Jan-Oct training | 48000 | 677 | 340 | 337 | 50.2216 | 49.7784 | 0 |
| Jan-Aug classifier training | 38477 | 533 | 267 | 266 | 50.0938 | 49.9062 | 0 |
| Sep-Oct classifier test | 9523 | 144 | 73 | 71 | 50.6944 | 49.3056 | 0 |
| latest-origin OOF | 19194 | 277 | 142 | 135 | 51.2635 | 48.7365 | 0 |

## 3. quote_signal by month

Pearson correlations; sample SD; raw market correlation uses observed pairs; clean market correlation uses same-date imputation.

| month | n | quote_mean | quote_std_sample | corr_quote_rpm | corr_quote_market_raw | corr_quote_market_clean | proxy_MAE_dollars | proxy_MAPE_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-01 | 4918 | 2.0684 | 0.2503 | 0.4467 | 0.0839 | 0.0821 | 75.3479 | 3.4926 |
| 2025-02 | 4337 | 2.0971 | 0.2517 | 0.4790 | 0.1008 | 0.0983 | 55.2426 | 2.8481 |
| 2025-03 | 5036 | 2.1795 | 0.2729 | 0.4858 | 0.1128 | 0.1142 | 63.8565 | 3.1643 |
| 2025-04 | 4819 | 1.9634 | 0.2604 | -0.4717 | -0.1331 | -0.1314 | 403.3130 | 20.1014 |
| 2025-05 | 4913 | 1.9128 | 0.2658 | -0.4592 | -0.0716 | -0.0688 | 403.3986 | 20.9965 |
| 2025-06 | 4783 | 2.2960 | 0.2889 | 0.4766 | 0.0518 | 0.0526 | 82.8454 | 2.8107 |
| 2025-07 | 4912 | 1.9230 | 0.2651 | -0.4683 | -0.1474 | -0.1504 | 415.7703 | 20.6781 |
| 2025-08 | 4759 | 2.0522 | 0.2227 | -0.0136 | 0.0208 | 0.0213 | 317.2564 | 14.7432 |
| 2025-09 | 4670 | 2.1998 | 0.2799 | 0.4814 | 0.0315 | 0.0323 | 71.7493 | 2.8788 |
| 2025-10 | 4853 | 1.9428 | 0.2692 | -0.4340 | -0.0624 | -0.0631 | 413.7224 | 20.9937 |
| 2025-11 | 5836 | 2.0553 | 0.2203 | NOT AVAILABLE | -0.0375 | -0.0347 | NOT AVAILABLE | NOT AVAILABLE |
| 2025-12 | 6164 | 2.0476 | 0.2193 | NOT AVAILABLE | -0.0235 | -0.0224 | NOT AVAILABLE | NOT AVAILABLE |

November–December target correlation and proxy errors: NOT AVAILABLE; validation.csv has no posted_rate.

## 4. Non-flagged OOF residuals by month

Absolute-error percentiles: dollars; linear quantile interpolation. Non-flagged: existing retrospective OOF flags.

| month | n | mean_signed_pct_error | MAPE_pct | abs_error_p90_dollars | abs_error_p99_dollars |
| --- | --- | --- | --- | --- | --- |
| 2025-07 | 4847 | -0.5339 | 1.5367 | 88.9864 | 182.5634 |
| 2025-08 | 4691 | 0.3289 | 1.4059 | 76.3455 | 168.5618 |
| 2025-09 | 4606 | -2.2539 | 2.5980 | 141.4115 | 326.5113 |
| 2025-10 | 4773 | -1.7106 | 2.0328 | 104.9656 | 213.3340 |
| overall OOF | 18917 | -1.0356 | 1.8878 | 103.2050 | 231.8920 |

January–June OOF errors: NOT AVAILABLE; no saved held-out predictions for those months.

## File checks

| protected_file_count | changed_protected_files | OOF_rows | OOF_latest_origin_verified | OOF_first_date | OOF_last_date | market_coefficient | seed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 95 | 0 | 19194 | True | 2025-07-01 | 2025-10-31 | 0.1194 | 42 |

Added: audit/residual_tail_audit.py; audit/residual_tail/ CSVs, protocols, checks and REPORT.md.
