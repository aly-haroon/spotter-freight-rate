"""Read-only production audit. Run from root: python -B -m audit.run_audit.

All writes stay in audit/. Existing fitting code is imported, never changed.
"""
import contextlib
import hashlib
import io
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost as xgb
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.config import FOLDS, SEED
from src.data import clean, flag_outliers, load
from src.features import base_features
from src.model import GBM_PARAMS, TwoStageModel
from src.predict import build_december_features, city_coordinates
from src.validate import BaselineModel, metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(actual, predicted):
    a, p = np.asarray(actual), np.asarray(predicted)
    return {"n": len(a), **metrics(a, p),
            "MedAE": float(np.median(np.abs(p-a))), "ME": float(np.mean(p-a))}


def save(name, rows):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(OUT / f"{name}.csv", index=False)
    return df


def main():
    OUT.mkdir(exist_ok=True)
    protected = sorted(set(list((ROOT / "src").glob("*.py")) + list((ROOT / "data").glob("*.csv"))
                           + list((ROOT / "reports").rglob("*.*")) + list((ROOT / "submission_v1").glob("*.*"))
                           + list(ROOT.glob("*.csv")) + [ROOT / "score.py", ROOT / "requirements-lock.txt"]))
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected if p.is_file()}
    (OUT / "protected_hashes_before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    raw = load(ROOT / "data/train_test.csv")
    final_raw = load(ROOT / "data/validation.csv")
    data = clean(raw)  # exactly as src.validate.main, before the time splits
    final = clean(final_raw)
    facts = {
        "train_rows": len(raw), "train_start": str(raw.date.min().date()), "train_end": str(raw.date.max().date()),
        "final_rows": len(final_raw), "final_start": str(final_raw.date.min().date()), "final_end": str(final_raw.date.max().date()),
        "train_unique_ids": bool(raw.load_id.is_unique), "final_unique_ids": bool(final_raw.load_id.is_unique),
        "id_overlap": len(set(raw.load_id) & set(final_raw.load_id)),
        "final_has_target": "posted_rate" in final_raw,
        "notebooks": [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.ipynb") if ".venv" not in p.parts],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__, "xgboost": xgb.__version__},
        "source_copy_hashes_equal": {
            "train": (digest(ROOT / "train-test.csv") == digest(ROOT / "data/train_test.csv")) if (ROOT / "train-test.csv").exists() else "NOT AVAILABLE: optional original root copy not retained",
            "final": (digest(ROOT / "validation.csv") == digest(ROOT / "data/validation.csv")) if (ROOT / "validation.csv").exists() else "NOT AVAILABLE: optional original root copy not retained"},
    }
    for name, df in [("train", raw), ("final", final_raw)]:
        facts[name + "_quality"] = {
            "weight_missing": int(df.weight.isna().sum()), "weight_negative": int(df.weight.lt(0).sum()),
            "market_missing": int(df.market_index.isna().sum()),
            "all_missing_market_days": int(df.groupby("date").market_index.count().eq(0).sum()),
            "nonpositive_distance": int(df.distance.le(0).sum()),
            "nonpositive_market_observed": int(df.market_index.le(0).sum())}
    train_cities = set(raw.pickup) | set(raw.delivery)
    final_cities = set(final_raw.pickup) | set(final_raw.delivery)
    new_cities = sorted(final_cities - train_cities)
    lanes = set(zip(raw.pickup, raw.delivery))
    facts.update(new_cities=new_cities,
                 final_rows_new_city=int((final_raw.pickup.isin(new_cities) | final_raw.delivery.isin(new_cities)).sum()),
                 final_rows_unseen_route=int(sum((p,d) not in lanes for p,d in zip(final_raw.pickup, final_raw.delivery))),
                 base_feature_names=list(base_features(data).columns), gbm_params=GBM_PARAMS,
                 training_outliers=int(flag_outliers(data).sum()))
    coordinates = pd.concat([
        df[[side, side+"_lat", side+"_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        for df in [raw, final_raw] for side in ["pickup", "delivery"]])
    facts["cities_with_conflicting_coordinates"] = int(coordinates.drop_duplicates().groupby("city").size().gt(1).sum())

    fold_rows, frames = [], []
    saved_metrics = pd.read_csv(ROOT / "reports/backtest_results.csv")
    for fold, start, end in FOLDS:
        train_full = data.loc[data.date < start]
        train = train_full.loc[~flag_outliers(train_full)]
        test = data.loc[data.date.ge(start) & data.date.lt(end)]
        saved = pd.read_csv(ROOT / f"reports/tuning_experiment/predictions_{fold}.csv").set_index("load_id")
        model_pred = saved.loc[test.load_id, "prediction_depth6_lr003"].to_numpy()
        assert np.array_equal(saved.loc[test.load_id, "posted_rate"].to_numpy(), test.posted_rate.to_numpy())
        baseline = BaselineModel().fit(train).predict(test)
        frame = test.copy()
        frame["fold"] = fold
        frame["forecast_origin"] = pd.Timestamp(start)
        frame["model_pred"] = model_pred
        frame["baseline_pred"] = baseline
        frame["diagnostic_outlier"] = flag_outliers(test)
        seen = set(zip(train_full.pickup, train_full.delivery))
        kept_seen = set(zip(train.pickup, train.delivery))
        frame["route_status"] = ["seen" if (p,d) in seen else "unseen" for p,d in zip(test.pickup,test.delivery)]
        frame["route_status_fit_rows"] = ["seen" if (p,d) in kept_seen else "unseen" for p,d in zip(test.pickup,test.delivery)]
        for name, pred, label in [("baseline", baseline, "1 baseline"), ("model", model_pred, "5 CHOSEN")]:
            stats = scores(test.posted_rate, pred)
            prior = saved_metrics.loc[saved_metrics.fold.eq(fold) & saved_metrics.segment.eq("all rows")
                                      & saved_metrics.variant.str.startswith(label)].iloc[0]
            for metric in ["MAE", "RMSE", "MAPE%"]:
                assert abs(stats[metric] - prior[metric]) < 1e-8
            fold_rows.append({"fold": fold, "model": name,
                              "train_start": str(train_full.date.min().date()), "train_end": str(train_full.date.max().date()),
                              "test_start": str(test.date.min().date()), "test_end": str(test.date.max().date()),
                              "train_raw_n": len(train_full), "train_fit_n": len(train), **stats})
        frames.append(frame)
        print(f"Saved predictions and baseline verified: {fold}", flush=True)
    folds = save("per_fold", fold_rows)
    occurrences = pd.concat(frames, ignore_index=True)
    unique = occurrences.sort_values(["forecast_origin", "load_id"]).drop_duplicates("load_id", keep="last").copy()
    save("oof_predictions", unique)
    facts["oof_prediction_occurrences"] = len(occurrences)
    facts["oof_unique_rows_latest_origin"] = len(unique)
    facts["oof_duplicate_occurrences"] = len(occurrences)-len(unique)
    save("oof_summary", [{"aggregation": scope, "model": name, **scores(df.posted_rate, df[col])}
                         for scope, df in [("all fold occurrences", occurrences), ("unique latest-origin", unique)]
                         for name, col in [("baseline", "baseline_pred"), ("model", "model_pred")]])
    save("monthly", [{"month": str(month), "model": name, **scores(df.posted_rate, df[col])}
                     for month, df in unique.groupby(unique.date.dt.to_period("M"))
                     for name, col in [("baseline", "baseline_pred"), ("model", "model_pred")]])
    unique["distance_bucket"] = pd.cut(unique.distance, [-np.inf,250,500,1000,2000,np.inf],
        labels=["<=250", "(250,500]", "(500,1000]", "(1000,2000]", ">2000"])
    unique["price_bucket"] = pd.cut(unique.posted_rate, [-np.inf,1000,2000,4000,np.inf], right=False,
        labels=["<1000", "[1000,2000)", "[2000,4000)", ">=4000"])
    unique["weight_bucket"] = pd.cut(unique.weight, [-np.inf,10000,25000,40000,np.inf],
        labels=["<=10000", "(10000,25000]", "(25000,40000]", ">40000"]).astype(object).fillna("missing")
    unique["weight_quality"] = np.select([unique.weight_missing.eq(1),unique.weight_was_negative.eq(1)],
        ["originally missing", "originally negative"], default="originally nonnegative")
    breakdowns = []
    for scope, df in [("F3", unique.loc[unique.fold.eq("F3")]), ("unique OOF", unique)]:
        for dimension in ["equipment", "distance_bucket", "price_bucket", "weight_bucket", "weight_quality", "route_status"]:
            for bucket, group in df.groupby(dimension, observed=True):
                row = {"scope": scope, "dimension": dimension, "bucket": bucket, "n": len(group)}
                for name, col in [("baseline", "baseline_pred"), ("model", "model_pred")]:
                    stats = scores(group.posted_rate, group[col])
                    row.update({name+"_MAE": stats["MAE"], name+"_MAPE%": stats["MAPE%"]})
                breakdowns.append(row)
    save("breakdowns", breakdowns)
    save("outlier_diagnostic", [{"scope": scope, "flagged": bool(flag), "model": name, **scores(g.posted_rate,g[col])}
        for scope, df in [("F3",unique.loc[unique.fold.eq("F3")]),("unique OOF",unique)]
        for flag, g in df.groupby("diagnostic_outlier")
        for name,col in [("baseline","baseline_pred"),("model","model_pred")]])

    # Exact current F3 fit, plus December-like replacement, without writing submissions.
    print("Fitting unchanged F3 model for reproducibility and December simulation...", flush=True)
    train3 = data.loc[data.date.lt("2025-09-01")]
    train3 = train3.loc[~flag_outliers(train3)]
    test3 = data.loc[data.date.ge("2025-09-01")]
    m3 = TwoStageModel().fit(train3)
    actual3 = m3.predict(test3)
    prior3 = occurrences.loc[occurrences.fold.eq("F3")].set_index("load_id").loc[test3.load_id,"model_pred"].to_numpy()
    facts["f3_refit_max_abs_difference"] = float(np.max(np.abs(actual3-prior3)))
    approximation = test3.copy()
    approximation["market_index"] = approximation.groupby("date").market_index.transform("mean")
    approximation["quote_signal"] = np.nan
    simulated = m3.predict(approximation)
    sim_rows = []
    route_mask = test3.pickup.eq("Lexington") & test3.delivery.eq("Fort Wayne")
    for scope, mask in [("all F3",np.ones(len(test3),dtype=bool)),("Lexington to Fort Wayne F3",route_mask.to_numpy())]:
        if mask.any():
            for name,pred in [("true row features",actual3),("December-like daily market mean; quote missing",simulated)]:
                sim_rows.append({"scope":scope,"features":name,**scores(test3.posted_rate.to_numpy()[mask],pred[mask])})
    save("december_simulation",sim_rows)
    facts["f3_market_coefficient"] = m3.market_.b_market_

    # Additional fixed three-city diagnostic requested by the audit.
    past_raw = raw.loc[raw.date.lt("2025-09-01")]
    city_pool = sorted(set(past_raw.pickup)|set(past_raw.delivery))
    cities3 = sorted(np.random.default_rng(SEED).choice(city_pool,3,replace=False).tolist())
    hidden_train = past_raw.loc[~(past_raw.pickup.isin(cities3)|past_raw.delivery.isin(cities3))]
    restricted = clean(hidden_train)
    restricted = restricted.loc[~flag_outliers(restricted)]
    mask = (test3.pickup.isin(cities3)|test3.delivery.isin(cities3)).to_numpy()
    (OUT / "three_city_protocol.json").write_text(json.dumps({"seed":SEED,"cities":cities3,
        "selection":"sample 3 without replacement from sorted Jan-Aug city names, before audit fit",
        "train_raw_n":len(hidden_train),"train_fit_n":len(restricted),"test_touching_cities_n":int(mask.sum())},indent=2),encoding="utf-8")
    print("Fitting three-city exclusion diagnostic:",cities3,flush=True)
    restricted_pred = TwoStageModel().fit(restricted).predict(test3)
    save("three_city_results",[{"model":name,"training_rows":n,**scores(test3.posted_rate.to_numpy()[mask],pred[mask])}
        for name,n,pred in [("reference",len(train3),actual3),("3 cities withheld",len(restricted),restricted_pred)]])

    print("Refitting unchanged final model twice in memory; no production writes...",flush=True)
    train_all = data.loc[~flag_outliers(data)]
    fitted = TwoStageModel().fit(train_all)
    p1 = fitted.predict(final)
    submitted = pd.read_csv(ROOT / "validation_predictions.csv")
    by_id = pd.Series(p1,index=final.load_id).reindex(submitted.load_id).round(2).to_numpy()
    facts["final_saved_prediction_max_abs_difference_rounded"] = float(np.max(np.abs(by_id-submitted.predicted_rate)))
    p2 = TwoStageModel().fit(train_all).predict(final)
    facts["two_final_refits_max_abs_difference_unrounded"] = float(np.max(np.abs(p1-p2)))
    facts["final_market_coefficient"] = fitted.market_.b_market_
    december = pd.read_csv(ROOT / "data/december_chart_inputs.csv")
    coords = city_coordinates([train_all,final])
    december_features = build_december_features(december,final,coords)
    save("december_features",december_features)
    december_refit = fitted.predict(december_features).round(2)
    facts["december_saved_prediction_max_abs_difference_rounded"] = float(np.max(np.abs(december_refit-december.predicted_rate)))
    facts["december_coordinates"] = coords.loc[["Lexington","Fort Wayne"]].reset_index().to_dict("records")
    save("december_market_sources",final_raw.groupby("date").agg(
        rows=("load_id","size"),observed_market_n=("market_index","count"),mean_market=("market_index","mean")
        ).loc["2025-12-01":"2025-12-31"].reset_index())

    # Adversarial classifier: hold out complete dates, exclude date, IDs and labels.
    print("Running adversarial validation on held-out dates...",flush=True)
    features = pd.concat([base_features(data),base_features(final)],ignore_index=True)
    features["log_market_index"] = np.log(pd.concat([data.market_index,final.market_index],ignore_index=True))
    sources = np.r_[np.zeros(len(data),dtype=int),np.ones(len(final),dtype=int)]
    dates = pd.concat([data.date,final.date],ignore_index=True)
    fit_idx,test_idx = next(GroupShuffleSplit(n_splits=1,test_size=0.30,random_state=SEED).split(features,sources,groups=dates))
    adversarial, importance = [], []
    for feature_set in ["deployed features", "deployed plus unused quote_signal"]:
        x = features.copy()
        if feature_set.endswith("quote_signal"):
            x["quote_signal"] = pd.concat([data.quote_signal,final.quote_signal],ignore_index=True)
        classifier = xgb.XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,
            subsample=0.8,colsample_bytree=0.8,tree_method="hist",random_state=SEED,n_jobs=4,eval_metric="logloss")
        classifier.fit(x.iloc[fit_idx],sources[fit_idx])
        held = x.iloc[test_idx].copy()
        auc = roc_auc_score(sources[test_idx],classifier.predict_proba(held)[:,1])
        adversarial.append({"features":feature_set,"AUC":auc,"classifier_train_n":len(fit_idx),"classifier_test_n":len(test_idx),
            "train_source_in_classifier_test":int((sources[test_idx]==0).sum()),"final_source_in_classifier_test":int((sources[test_idx]==1).sum()),
            "training_dates":int(dates.iloc[fit_idx].nunique()),"test_dates":int(dates.iloc[test_idx].nunique())})
        rng = np.random.default_rng(SEED)
        for col in x.columns:
            drops = []
            for _ in range(3):
                permuted = held.copy()
                permuted[col] = rng.permutation(permuted[col].to_numpy())
                drops.append(auc-roc_auc_score(sources[test_idx],classifier.predict_proba(permuted)[:,1]))
            importance.append({"features":feature_set,"feature":col,"heldout_permutation_AUC_drop":float(np.mean(drops)),
                               "normalized_gain":float(classifier.feature_importances_[x.columns.get_loc(col)])})
    save("adversarial",adversarial)
    save("adversarial_importance",pd.DataFrame(importance).sort_values(["features","heldout_permutation_AUC_drop"],ascending=[True,False]))

    distribution=[]
    periods=[("Sep-Oct",raw.loc[raw.date.ge("2025-09-01")]),("Nov",final_raw.loc[final_raw.date.dt.month.eq(11)]),
             ("Dec",final_raw.loc[final_raw.date.dt.month.eq(12)]),("Nov-Dec",final_raw)]
    for col in ["market_index","quote_signal"]:
        ref=periods[0][1][col].dropna()
        for name,df in periods:
            values=df[col].dropna()
            distribution.append({"feature":col,"period":name,"n_observed":len(values),"n_missing":int(df[col].isna().sum()),
                "mean":values.mean(),"std":values.std(),"min":values.min(),"p05":values.quantile(.05),
                "median":values.median(),"p95":values.quantile(.95),"max":values.max(),
                "KS_vs_SepOct":float(ks_2samp(ref,values).statistic)})
    save("distribution_shift",distribution)
    save("quote_proxy",[{"month":str(month),**scores(g.posted_rate,g.quote_signal*g.distance),
                          "quote_rpm_correlation":g.quote_signal.corr(g.posted_rate/g.distance)}
         for month,g in raw.groupby(raw.date.dt.to_period("M"))])

    hist=raw.loc[raw.pickup.eq("Lexington") & raw.delivery.eq("Fort Wayne")].copy()
    hist["rpm"]=hist.posted_rate/hist.distance
    save("historical_chart_route",hist)
    route_summary=[]
    for name,df in [("all historical route rows",hist),("historical route Dry Van",hist.loc[hist.equipment.eq("Dry Van")])]:
        route_summary.append({"scope":name,"n":len(df),"price_min":df.posted_rate.min(),"price_median":df.posted_rate.median(),
            "price_max":df.posted_rate.max(),"rpm_min":df.rpm.min(),"rpm_median":df.rpm.median(),"rpm_max":df.rpm.max(),
            "distance_min":df.distance.min(),"distance_max":df.distance.max(),"weight_min":df.weight.min(),"weight_max":df.weight.max()})
    route_summary.append({"scope":"December fixed chart","n":len(december),"price_min":december.predicted_rate.min(),
        "price_median":december.predicted_rate.median(),"price_max":december.predicted_rate.max(),
        "rpm_min":(december.predicted_rate/december.distance).min(),"rpm_median":(december.predicted_rate/december.distance).median(),
        "rpm_max":(december.predicted_rate/december.distance).max(),"distance_min":december.distance.min(),"distance_max":december.distance.max(),
        "weight_min":december.weight.min(),"weight_max":december.weight.max()})
    save("chart_route_comparison",route_summary)
    sorted_dec=december.sort_values("date").copy()
    diffs=sorted_dec.predicted_rate.diff().dropna()
    facts["december_shape"]={"first":float(sorted_dec.predicted_rate.iloc[0]),"last":float(sorted_dec.predicted_rate.iloc[-1]),
        "day_change_min":float(diffs.min()),"day_change_max":float(diffs.max()),"median_abs_day_change":float(diffs.abs().median()),
        "linear_slope_dollars_per_day":float(np.polyfit(np.arange(len(sorted_dec)),sorted_dec.predicted_rate,1)[0])}
    sorted_dec["weekday"]=pd.to_datetime(sorted_dec.date).dt.day_name()
    save("december_weekday",sorted_dec.groupby("weekday",sort=False).predicted_rate.agg(["count","mean","min","max"]).reset_index())
    # Run provided scorer with all generated outputs redirected into audit/.
    result=subprocess.run([sys.executable,"-B",str(ROOT/"score.py"),"--predictions",str(ROOT/"validation_predictions.csv"),
        "--december-predictions",str(ROOT/"data/december_chart_inputs.csv"),"--output-dir",str(OUT/"scorer_results")],
        capture_output=True,text=True,cwd=ROOT)
    (OUT/"scorer.log").write_text(result.stdout+result.stderr,encoding="utf-8")
    facts["scorer_returncode"]=result.returncode
    expected={f"TE-{i:06d}" for i in range(1,12001)}
    facts["submission_checks"]={"columns":list(submitted.columns),"rows":len(submitted),
        "unique_ids":bool(submitted.load_id.is_unique),"ids_exact_expected":set(submitted.load_id)==expected,
        "finite_positive":bool(np.isfinite(submitted.predicted_rate).all() and submitted.predicted_rate.gt(0).all()),
        "same_as_backup":(digest(ROOT/"validation_predictions.csv")==digest(ROOT/"submission_v1/validation_predictions.csv")) if (ROOT/"submission_v1/validation_predictions.csv").exists() else "NOT AVAILABLE: optional local backup not retained"}
    save("output_distributions",[{"file":name,"n":len(values),"min":values.min(),"median":values.median(),"max":values.max()}
        for name,values in [("validation_predictions.csv",submitted.predicted_rate),("data/december_chart_inputs.csv",december.predicted_rate)]])
    after={str(p.relative_to(ROOT)):digest(p) for p in protected if p.is_file()}
    facts["protected_files_unchanged"] = before==after
    facts["protected_file_count"]=len(before)
    (OUT/"facts.json").write_text(json.dumps(facts,indent=2,default=str),encoding="utf-8")
    assert before==after, "A protected file changed during audit"
    print("Audit diagnostics complete. All protected source, data, saved results and submissions unchanged.",flush=True)


if __name__=="__main__":
    main()
