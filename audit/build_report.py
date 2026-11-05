"""Build a report from audit outputs; no model fitting or production writes."""
import json
from pathlib import Path

import pandas as pd
from audit.run_audit import scores

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
facts = json.loads((OUT / "facts.json").read_text())


def read(name):
    return pd.read_csv(OUT / (name + ".csv"))


def table(df, digits=2):
    def fmt(v):
        if pd.isna(v):
            return "NOT AVAILABLE"
        if isinstance(v, float):
            return f"{v:,.{digits}f}"
        return str(v).replace("|", "/")
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |",
        "| " + " | ".join(["---"]*len(df.columns)) + " |"] +
        ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)])


def main():
    sections = []
    def add(text):
        sections.append(text.strip())

    add("""# Spotter freight-rate solution audit

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
.\\.venv\\Scripts\\python.exe -B -u -m audit.run_audit
.\\.venv\\Scripts\\python.exe -B -m audit.build_report
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
""")
    pf = read("per_fold")
    ftable = pf.copy()
    ftable["Train dates (2025)"] = ftable.train_start.str[5:] + " to " + ftable.train_end.str[5:]
    ftable["Test dates (2025)"] = ftable.test_start.str[5:] + " to " + ftable.test_end.str[5:]
    ftable["Train raw -> fitted"] = ftable.train_raw_n.astype(str) + " -> " + ftable.train_fit_n.astype(str)
    ftable.loc[ftable.fold.eq("F3"),"fold"] = "F3 (MOST RECENT)"
    add("## 2. PER-FOLD RESULTS\n\n" + table(ftable[["fold","model","Train dates (2025)","Test dates (2025)",
        "Train raw -> fitted","n","MAE","RMSE","MAPE%","MedAE"]]))
    means = pf.groupby("model",sort=False)[["MAE","RMSE","MAPE%","MedAE"]].mean().reset_index()
    add("Reported headline = arithmetic mean of the three fold scores, each fold given equal weight. "
        "The RMSE below is the mean of fold RMSEs, not a pooled RMSE; the MedAE below is the mean of fold medians.\n\n" + table(means))
    add("For monthly and overall breakdowns, use each shipment's latest available forecast "
        "origin, chosen by date rather than error. There are 28,623 fold prediction "
        "occurrences, 19,194 unique shipments and 9,429 repeated occurrences. "
        "This retains F1 July, F2 August and F3 September-October. Thus the first "
        "three months use first-month forecasts; October uses second-month forecasts. "
        "The audit also reports pooling all occurrences, which double-counts some shipments.\n\n" + table(read("oof_summary")))
    add("## 3. ERROR OVER TIME\n\nUnique latest-origin forecasts; ME is signed dollars.\n\n" +
        table(read("monthly")[["month","model","n","MAE","MAPE%","ME"]]) +
        "\n\nPercentage error rises later in the period; dollar MAE is not monotonically "
        "increasing. Both recent months show systematic underprediction. The effect "
        "is not solely extreme prices: F3's non-flagged diagnostic subset has mean "
        "signed error -$45.61 (versus -$79.23 across all F3 rows). These observations "
        "do not establish what November-December accuracy will be.")
    b = read("breakdowns")
    add("""## 4. ERROR BREAKDOWNS

All rows included. Price buckets use actual posted_rate (retrospective diagnosis
only); distance is miles and weight is pounds after absolute-value cleaning.
Missing weights are separate. A route is the ordered pickup-delivery pair,
regardless of equipment; 'seen' means present in raw training rows before outlier
exclusion, matching the original backtest definition. Small buckets are unstable.
""")
    for scope in ["F3","unique OOF"]:
        t=b.loc[b.scope.eq(scope),["dimension","bucket","n","model_MAE","model_MAPE%","baseline_MAE","baseline_MAPE%"]]
        add("### " + ("Most recent fold: September-October" if scope=="F3" else "Overall unique out-of-fold") + "\n\n"+table(t))
    p3=json.loads((OUT/"three_city_protocol.json").read_text())
    oof=read("oof_predictions")
    strict=pd.DataFrame([{"scope":scope,"route status after filtering":status,
                          **scores(g.posted_rate,g.model_pred)}
        for scope,df in [("F3",oof.loc[oof.fold.eq("F3")]),("unique OOF",oof)]
        for status,g in df.groupby("route_status_fit_rows")])
    strict.to_csv(OUT/"routes_after_filtering.csv",index=False)
    add("## 5. UNSEEN CITIES AND ROUTES\n\nRoute errors are in section 4; F3 has only "
        "21 unseen-route examples and overall unique OOF has 68. Their lower error "
        "does not prove unseen routes are easier. Final data has 1,447 rows touching "
        "a new city and 1,461 on unseen ordered routes. New cities: " + ", ".join(facts["new_cities"]) + ".\n\n"
        "One OOF shipment's route appears in raw training but disappears after outlier "
        "filtering. If 'seen' instead requires a row in the actual fitted training "
        "data, model results are:\n\n"+table(strict[["scope","route status after filtering","n","MAE","MAPE%"]])+"\n\n"
        "**New three-city diagnostic:** hide " + ", ".join(p3["cities"]) +
        " from both pickup and delivery in January-August, then score the same "
        "999 September-October rows touching those cities. Cities were sampled "
        "before fitting, from sorted historical city names with seed 42. Restricted "
        "training has 34,581 raw rows and 34,091 after outlier exclusion.\n\n" + table(read("three_city_results")) +
        "\n\nExisting eight-city diagnostic is also retained as separate evidence. Hidden "
        "cities: Bakersfield, Hartford, Memphis, New York, San Antonio, Toledo, Tulsa, "
        "Washington. On the same selected-city test rows:\n\n" + table(pd.read_csv(ROOT/"reports/city_holdout/results.csv").loc[
            lambda d:d.segment.eq("all rows, hidden cities"),["model","training_rows","n","MAE","RMSE","MAPE%"]]) +
        "\n\nBoth tests show a modest error increase. Removing cities also removes training "
        "rows; these single city selections do not isolate geography from sample size "
        "and do not guarantee accuracy on the eight actual final-only cities.")
    add(f"""## 6. LEAKAGE AND REPRODUCIBILITY AUDIT

| Question | Verdict | Evidence / limitation |
| --- | --- | --- |
| Are all imputations fit only on fold training rows? | NO | `clean` computes same-date market means from the frame being cleaned. `src.validate.main` cleans the whole historical frame before splitting; dates do not cross fold boundaries, so no future date enters a training-date mean. Test-date means still use test-batch features. This is batch-dependent preprocessing, not target leakage. |
| Are active fitted model/encoding statistics training-only? | YES | Training outlier references, market slope, recent level and trees use the fold's training rows. Final premiums are off; optional premium tables are fold-training-only and OOF inside that training data. No scaler or fitted weight imputer exists. |
| Are date-trend, load_id or leakage-prone inputs used? | YES, with qualification | Date is used for historical weeks/recent level and same-date market imputation; final linear trend is off. `t_months` is computed but unused by the selected model. `load_id` is only an output/join key. `quote_signal` is excluded. Availability/provenance of future market_index is not independently verified. |
| Are any validation.csv labels used? | NO | No posted_rate column exists. Only final feature values are used, including date-level market means and city coordinates for December. |
| Are test labels used anywhere? | YES | Test prices are used for scoring, outlier diagnostic flags and selection among development variants. They are not model prediction inputs. Filtered scores therefore must not be presented as all-load performance. |
| Can final fitting reproduce saved predictions with fixed seeds? | YES on this environment | Two fresh final fits had maximum unrounded prediction difference ${facts['two_final_refits_max_abs_difference_unrounded']:.2f}. Refit predictions rounded to cents matched every saved final and December prediction exactly. F3 refit maximum difference from saved CSV floats was {facts['f3_refit_max_abs_difference']:.3g} dollars. |
| Is a completely fresh-machine rebuild verified? | UNCLEAR | Audit reused the installed environment. A lock file exists, but a new environment, hardware and OS were not tested. |

Runtime: Python {facts['versions']['python']}, NumPy {facts['versions']['numpy']},
pandas {facts['versions']['pandas']}, scikit-learn {facts['versions']['sklearn']},
XGBoost {facts['versions']['xgboost']}, SciPy {facts['versions']['scipy']}.
No production training command was invoked because it would overwrite submissions;
the audit imported and fitted its unchanged classes and reconstructed its output
mapping and rounding in memory. No serialized fitted model was found among the
project files; reproducibility relies on the code, inputs and environment.
""")
    add("""## 7. DISTRIBUTION SHIFT

Adversarial validation means predicting whether a row comes from labeled history
or final data. AUC near 0.5 means little separability; a higher value means the
feature distributions are distinguishable, not necessarily that regression fails.

Audit classifier: XGBoost binary classifier, 300 trees, depth 3, learning rate
0.05, subsample and colsample 0.8, hist, seed 42, four threads. A single
GroupShuffleSplit holds out 30% of complete dates, seed 42; no date is in both
classifier partitions. Raw date, load ID, target and target-derived fields are
excluded. 'Deployed features' means the 12 tree inputs plus log(market_index).
The second variant adds the excluded quote_signal to diagnose its shift.
""")
    add(table(read("adversarial"),4))
    imp=read("adversarial_importance").groupby("features",sort=False).head(3)
    add("Top features by held-out permutation importance, averaged over three shuffles. "
        "AUC drop measures how much separability is lost after shuffling that feature; "
        "correlated features can share importance.\n\n" + table(imp[["features","feature","heldout_permutation_AUC_drop"]],4))
    add("Raw observed distributions below exclude missing values. SD is sample standard "
        "deviation; KS is the largest distribution-function difference relative to "
        "September-October (not a p-value).\n\n" + table(read("distribution_shift")[["feature","period","n_observed","n_missing","mean","std","p05","median","p95","KS_vs_SepOct"]],4))
    proxy=read("quote_proxy").loc[lambda d:d.month.isin(["2025-02","2025-10"]),["month","n","MAE","MAPE%","quote_rpm_correlation"]]
    add("The strong adversarial result compares all January-October history against "
        "November-December. Marginal market distributions are closer when comparing "
        "only September-October against the final period. Neither check measures "
        "unobserved final target relationships. Verified quote_signal x distance proxy:\n\n"+table(proxy,4)+
        "\n\nThe historical quote-price association changes sign. With no final labels, "
        "its final relationship to actual price is NOT AVAILABLE; a feature-only "
        "proxy cannot prove the signal is independent of the final target.")
    add("""## 8. DECEMBER PIPELINE

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
""")
    sim=read("december_simulation")
    add(table(sim[["scope","features","n","MAE","MAPE%"]],4))
    simdelta=[]
    for scope,g in sim.groupby("scope",sort=False):
        a=g.iloc[0];b_=g.iloc[1]
        simdelta.append({"scope":scope,"MAE change $":b_.MAE-a.MAE,"MAPE change percentage points":b_["MAPE%"]-a["MAPE%"]})
    add("Positive change means degradation; negative means improvement.\n\n"+table(pd.DataFrame(simdelta),4))
    add("Historical Lexington -> Fort Wayne comparison (rate per mile = total dollars / "
        "actual miles; rows differ in equipment, distance and weight):\n\n"+
        table(read("chart_route_comparison")[["scope","n","price_min","price_median","price_max","rpm_min","rpm_median","rpm_max"]],4)+
        "\n\nOnly 21 of the 32 historical route rows are Dry Van. Their distances span "
        "350.9-378.8 miles and weights 16,195-41,021 pounds, so they are not identical "
        "to the fixed chart shipment. Historical rows are in `audit/historical_chart_route.csv`.")
    shape=facts["december_shape"]
    add(f"""**Day-to-day shape:** a repeating weekly wave, with low Sunday/Monday and
high Wednesday/Thursday values, rather than abrupt isolated spikes. All tree
inputs are fixed, so all day-to-day movement comes from the daily market mean
through the market layer; weekday and holidays are not model features. The final
market exponent is {facts['final_market_coefficient']:.6f}. Adjacent-day changes range
from ${shape['day_change_min']:.2f} to +${shape['day_change_max']:.2f}; median absolute
daily change is ${shape['median_abs_day_change']:.3f}. December 1 is
${shape['first']:.2f} and December 31 is ${shape['last']:.2f}; the fitted linear slope
is only ${shape['linear_slope_dollars_per_day']:.4f}/day, so endpoint comparison
alone overstates the overall trend. No holiday-specific response is learned.
""")
    add("## 9. OUTPUT FILE CHECKS\n\n"+table(pd.DataFrame([
        {"Check":"validation_predictions.csv rows","Result":"PASS: 12,000"},
        {"Check":"Exact columns and order","Result":"PASS: load_id,predicted_rate"},
        {"Check":"IDs TE-000001 through TE-012000 once each","Result":"PASS: exact set; no duplicates or missing IDs"},
        {"Check":"All predictions finite and greater than zero","Result":"PASS"},
        {"Check":"December fixed rows, dates and seven-column structure","Result":"PASS: supplied score.py, exit code 0"},
        {"Check":"Prediction reproduction","Result":"PASS: exact saved values after cent rounding"},
        {"Check":"Production files changed by audit","Result":"NO: all 71 protected file hashes unchanged"}]))+
        "\n\nPrediction distributions (total dollars):\n\n"+table(read("output_distributions"),3)+
        "\n\nScorer rerun log: `audit/scorer.log`; its new chart is "
        "`audit/scorer_results/candidate_december.png`. The scorer checks file validity "
        "and draws the chart; it does not reveal hidden-label accuracy.")
    out=read("outlier_diagnostic")
    modeltail=out.loc[out.scope.eq("unique OOF") & out.model.eq("model")]
    flagged=modeltail.loc[modeltail.flagged].iloc[0]
    total=read("oof_summary").loc[lambda d:d.aggregation.eq("unique latest-origin") & d.model.eq("model")].iloc[0]
    contribution=flagged.n*flagged.MAE/(total.n*total.MAE)*100
    add("""## 10. TOP RISKS AND IMPROVEMENT IDEAS

These are ranked audit judgments, not additional model changes. Future improvement
estimates are NOT AVAILABLE without testing a specific intervention on new data.
The measured comparisons below are evidence, not promised gains.
""")
    risks=[
        {"Rank":1,"Weakness / evidence":"Time drift and underprediction: section 3 shows higher recent percentage errors and negative signed errors, including the non-flagged F3 subset.",
         "Improvement idea / measured scale":"Validate a separately specified level-calibration method on new future origins. The tested blend lowered all-row mean MAE by $2.28 and MAPE by 0.0670 points, but failed the preset F1 stability rule. Future gain: NOT AVAILABLE."},
        {"Rank":2,"Weakness / evidence":f"Extreme-price tail: {int(flagged.n)} of {int(total.n)} unique OOF rows were flagged, contributing {contribution:.2f}% of total absolute error; flagged-row model MAE was ${flagged.MAE:,.2f}.",
         "Improvement idea / measured scale":"Investigate price provenance and omitted pricing conditions; distinguish legitimate extremes from errors. Do not delete these rows from headline evaluation. Validated tail remedy and future gain estimate: NOT AVAILABLE."},
        {"Rank":3,"Weakness / evidence":"Future-feature availability and shift: December borrows same-date final-batch market information; adversarial separation is driven primarily by market_index.",
         "Improvement idea / measured scale":"Document the batch-conditional contract; separately evaluate a historical-only market forecast if needed. The within-holdout daily-mean approximation changed MAE by +$0.3685 and MAPE by +0.01321 points. Gain for a real advance market forecast: NOT AVAILABLE."},
        {"Rank":4,"Weakness / evidence":"Limited independent validation: overlapping development folds were reused for variant selection, and acceptance relied partly on a label-filtered diagnostic score. No untouched final labeled set exists.",
         "Improvement idea / measured scale":"Use new chronological origins or external labeled data for an independent evaluation; avoid claiming filtered error as full-population accuracy. Expected score correction or future improvement: NOT AVAILABLE."},
        {"Rank":5,"Weakness / evidence":"Sparse geography and chart evidence: very few naturally unseen historical routes, one three-city and one eight-city selection, and only six F3 rows on the chart route.",
         "Improvement idea / measured scale":"Repeat fixed city holdouts with size-matched controls before changing geographic modeling. Three-city withholding increased selected-row MAE by $2.03 and MAPE by 0.02189 points; the eight-city increase was $5.44 and 0.23586 points. Future gain: NOT AVAILABLE."},
    ]
    add(table(pd.DataFrame(risks)))
    add("**Diminishing returns from tree tuning:** the best depth-4 candidate changed "
        "all-row fold-mean MAE from $98.5519 to $98.3187 (a $0.2332 reduction) and "
        "MAPE from 4.271289% to 4.264583% (0.006706 percentage points). Its filtered "
        "improvement was 0.008108 points, below the preset 0.05-point minimum. It "
        "was faster and slightly better, so retaining depth 6 is a protocol decision, "
        "not proof that depth 6 has the lowest error. Both alternative recent windows "
        "were worse overall. Further blind tree/window tuning on the same folds has "
        "little supporting evidence; resolving drift, tail provenance and validation "
        "assumptions matters more. Keep the current submissions while reviewing these risks.")
    add("""## UNKNOWNS

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
""")
    (OUT/"AUDIT_REPORT.md").write_text("\n\n".join(sections)+"\n",encoding="utf-8")
    print("Wrote audit/AUDIT_REPORT.md")


if __name__=="__main__":
    main()
