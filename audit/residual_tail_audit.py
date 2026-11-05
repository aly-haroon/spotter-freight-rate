"""Additional diagnostics only. All writes stay in audit/residual_tail/.

Run from project root: python -B -u -m audit.residual_tail_audit
"""
import calendar
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import SEED
from src.data import DISTANCE_BANDS, clean, flag_outliers, load
from src.features import base_features
from src.model import MarketLayer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit/residual_tail"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def errors(frame):
    if frame.empty:
        return {"n": 0, "mean_signed_pct_error": np.nan, "MAPE_pct": np.nan,
                "abs_error_p90_dollars": np.nan, "abs_error_p99_dollars": np.nan}
    difference = frame.model_pred - frame.posted_rate
    relative = difference / frame.posted_rate
    return {"n": len(frame), "mean_signed_pct_error": 100 * relative.mean(),
            "MAPE_pct": 100 * relative.abs().mean(),
            "abs_error_p90_dollars": difference.abs().quantile(.90),
            "abs_error_p99_dollars": difference.abs().quantile(.99)}


def save(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / f"{name}.csv", index=False, na_rep="NOT AVAILABLE")
    return frame


def markdown(frame, precision=4):
    def value(v):
        if pd.isna(v):
            return "NOT AVAILABLE"
        if isinstance(v, (float, np.floating)):
            return f"{v:.{precision}f}"
        return str(v)
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
        ["| " + " | ".join(value(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = sorted(p for p in ROOT.rglob("*") if p.is_file()
                       and p.relative_to(ROOT).parts[0] not in {"audit", ".venv", ".git"})
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    raw = load(ROOT / "data/train_test.csv")
    final_raw = load(ROOT / "data/validation.csv")
    train = clean(raw)
    final = clean(final_raw)
    oof = pd.read_csv(ROOT / "audit/oof_predictions.csv", parse_dates=["date", "forecast_origin"])
    assert oof.load_id.is_unique
    assert oof.date.ge(oof.forecast_origin).all()
    assert np.isfinite(oof.model_pred).all()
    # Validate latest-origin selection against the saved per-fold predictions.
    candidates = []
    for fold, origin in [("F1", "2025-07-01"), ("F2", "2025-08-01"), ("F3", "2025-09-01")]:
        frame = pd.read_csv(ROOT / f"reports/tuning_experiment/predictions_{fold}.csv")
        frame["forecast_origin"] = pd.Timestamp(origin)
        candidates.append(frame)
    latest = pd.concat(candidates).sort_values("forecast_origin").drop_duplicates("load_id", keep="last").set_index("load_id")
    assert set(oof.load_id) == set(latest.index)
    assert np.array_equal(oof.forecast_origin.to_numpy(), latest.loc[oof.load_id, "forecast_origin"].to_numpy())
    np.testing.assert_allclose(oof.model_pred, latest.loc[oof.load_id, "prediction_depth6_lr003"], rtol=0, atol=1e-9)

    weekday = save("weekday_oof", [{"weekday": day, **errors(oof.loc[oof.date.dt.day_name().eq(day)])}
                                     for day in WEEKDAYS])
    memorial = max(pd.Timestamp(2025, 5, day) for day in range(1, 32)
                   if calendar.weekday(2025, 5, day) == calendar.MONDAY)
    labor = min(pd.Timestamp(2025, 9, day) for day in range(1, 31)
                if calendar.weekday(2025, 9, day) == calendar.MONDAY)
    holidays = [("Memorial Day", memorial), ("July 4", pd.Timestamp("2025-07-04")), ("Labor Day", labor)]
    windows, days = [], []
    for name, date in holidays:
        start, end = date-pd.Timedelta(days=3), date+pd.Timedelta(days=3)
        windows.append({"holiday": name, "holiday_date": str(date.date()), "start": str(start.date()),
                        "end": str(end.date()), **errors(oof.loc[oof.date.between(start, end)])})
        for offset in range(-3, 4):
            day = date+pd.Timedelta(days=offset)
            days.append({"holiday": name, "offset_days": offset, "date": str(day.date()),
                         **errors(oof.loc[oof.date.eq(day)])})
    holiday_windows = save("holiday_windows_oof", windows)
    holiday_days = save("holiday_days_oof", days)

    full_flag = flag_outliers(train)
    market = MarketLayer().fit(train.loc[~full_flag])
    adjusted = train.copy()
    adjusted["adjusted_rpm"] = adjusted.rpm / adjusted.market_index.pow(market.b_market_)
    adjusted_weekday = []
    for population, df in [("all training rows", adjusted), ("non-flagged training rows", adjusted.loc[~full_flag])]:
        overall_mean = df.adjusted_rpm.mean()
        for day in WEEKDAYS:
            group = df.loc[df.date.dt.day_name().eq(day)]
            adjusted_weekday.append({"population": population, "weekday": day, "n": len(group),
                "mean_rpm": group.rpm.mean(), "mean_market_adjusted_rpm": group.adjusted_rpm.mean(),
                "median_market_adjusted_rpm": group.adjusted_rpm.median(),
                "adjusted_mean_vs_population_mean_pct": 100*(group.adjusted_rpm.mean()/overall_mean-1)})
    adjusted_weekday = save("training_weekday_market_adjusted_rpm", adjusted_weekday)

    past = train.loc[train.date.lt("2025-09-01")]
    holdout = train.loc[train.date.ge("2025-09-01")]
    # Labels use the same retrospective comparable-load rule, separately in each split.
    y_train = flag_outliers(past).astype(int)
    y_test = flag_outliers(holdout).astype(int)
    x = base_features(train)
    x["log_market_index"] = np.log(train.market_index)
    x["quote_signal"] = train.quote_signal
    classifier_params = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=SEED, n_jobs=4, eval_metric="logloss")
    protocol = {"train_start": str(past.date.min().date()), "train_end": str(past.date.max().date()),
        "test_start": str(holdout.date.min().date()), "test_end": str(holdout.date.max().date()),
        "features": list(x.columns), "parameters": classifier_params,
        "label": "src.data.flag_outliers, evaluated separately on train and test; no outliers removed from classifier data",
        "selection": "single fixed time split and classifier; no tuning"}
    (OUT / "tail_classifier_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("Fitting fixed tail classifier: Jan-Aug -> Sep-Oct", flush=True)
    classifier = xgb.XGBClassifier(**classifier_params)
    classifier.fit(x.loc[past.index], y_train)
    probability = classifier.predict_proba(x.loc[holdout.index])[:, 1]
    classifier_result = save("tail_classifier", [{"train_start": protocol["train_start"], "train_end": protocol["train_end"],
        "test_start": protocol["test_start"], "test_end": protocol["test_end"],
        "train_n": len(past), "train_flagged_n": int(y_train.sum()),
        "test_n": len(holdout), "test_flagged_n": int(y_test.sum()),
        "test_prevalence_pct": 100*y_test.mean(), "AUC": roc_auc_score(y_test, probability),
        "average_precision": average_precision_score(y_test, probability)}])
    save("tail_classifier_predictions", pd.DataFrame({"load_id": holdout.load_id,
        "date": holdout.date, "flagged": y_test, "outlier_probability": probability}))

    tail_shares = []
    for population, df in [("Jan-Oct training", train), ("Jan-Aug classifier training", past),
                           ("Sep-Oct classifier test", holdout), ("latest-origin OOF", oof)]:
        band = pd.cut(df.distance, DISTANCE_BANDS)
        typical = df.groupby([df.equipment, band, df.date.dt.month], observed=True).rpm.transform("median")
        ratio = df.rpm/typical
        flagged = flag_outliers(df)
        n = int(flagged.sum())
        above = int((flagged & ratio.gt(1)).sum())
        below = int((flagged & ratio.lt(1)).sum())
        if population == "latest-origin OOF":
            assert np.array_equal(flagged.to_numpy(), df.diagnostic_outlier.to_numpy())
        tail_shares.append({"population": population, "rows": len(df), "flagged_n": n,
            "above_typical_n": above, "below_typical_n": below,
            "above_share_of_flagged_pct": 100*above/n, "below_share_of_flagged_pct": 100*below/n,
            "unclassified_flagged_n": n-above-below})
    tail_shares = save("tail_direction", tail_shares)

    monthly_signal = []
    combined = pd.concat([train, final], ignore_index=True)
    for date, group in combined.groupby(combined.date.dt.to_period("M")):
        has_labels = group.posted_rate.notna().all()
        signal_rpm_corr = group.quote_signal.corr(group.rpm) if has_labels else np.nan
        difference = group.quote_signal*group.distance-group.posted_rate
        monthly_signal.append({"month": str(date), "n": len(group), "quote_mean": group.quote_signal.mean(),
            "quote_std_sample": group.quote_signal.std(), "corr_quote_rpm": signal_rpm_corr,
            "corr_quote_market_clean": group.quote_signal.corr(group.market_index),
            "proxy_MAE_dollars": difference.abs().mean() if has_labels else np.nan,
            "proxy_MAPE_pct": 100*(difference.abs()/group.posted_rate).mean() if has_labels else np.nan})
    monthly_signal = save("quote_signal_monthly", monthly_signal)
    # Also retain correlation restricted to observed raw market_index pairs.
    raw_corr = pd.concat([raw, final_raw], ignore_index=True)
    raw_corr = save("quote_market_raw_correlation", [{"month": str(month),
        "observed_pair_n": int(g[["quote_signal", "market_index"]].notna().all(axis=1).sum()),
        "corr_quote_market_raw": g.quote_signal.corr(g.market_index)}
        for month, g in raw_corr.groupby(raw_corr.date.dt.to_period("M"))])
    monthly_signal = monthly_signal.merge(raw_corr, on="month", validate="one_to_one")
    save("quote_signal_monthly", monthly_signal)

    nonflag = oof.loc[~oof.diagnostic_outlier]
    monthly_nonflag = save("nonflag_monthly_residuals", [{"month": str(month), **errors(group)}
        for month, group in nonflag.groupby(nonflag.date.dt.to_period("M"))] + [{"month": "overall OOF", **errors(nonflag)}])
    after = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    assert before == after, "Protected files changed"
    checks = {"protected_file_count": len(before), "changed_protected_files": 0,
        "OOF_rows": len(oof), "OOF_latest_origin_verified": True,
        "OOF_first_date": str(oof.date.min().date()), "OOF_last_date": str(oof.date.max().date()),
        "market_coefficient": market.b_market_, "seed": SEED}
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    report = ["# Residual and tail diagnostics",
        "## 1. Weekday and holiday residuals",
        "Latest-origin OOF; all rows; signed % = 100 × mean((predicted − actual)/actual).",
        markdown(weekday[["weekday", "n", "mean_signed_pct_error", "MAPE_pct"]]),
        "Holiday windows: both endpoints included.",
        markdown(holiday_windows[["holiday", "holiday_date", "start", "end", "n", "mean_signed_pct_error", "MAPE_pct"]]),
        markdown(holiday_days[["holiday", "offset_days", "date", "n", "mean_signed_pct_error", "MAPE_pct"]]),
        "Memorial Day: NOT AVAILABLE; OOF begins 2025-07-01.",
        f"Market-adjusted RPM = posted_rate / distance / market_index**{market.b_market_:.12f}; training 2025-01-01–2025-10-31.",
        markdown(adjusted_weekday),
        "## 2. Tail predictability",
        "14 classifier inputs: 12 tree features + log(market_index) + quote_signal; seed 42; no gap; no flagged training rows excluded.",
        markdown(classifier_result, 6),
        "Typical price = equipment × distance-band × month median RPM × row distance; shares use flagged rows as denominator.",
        markdown(tail_shares),
        "## 3. quote_signal by month",
        "Pearson correlations; sample SD; raw market correlation uses observed pairs; clean market correlation uses same-date imputation.",
        markdown(monthly_signal[["month", "n", "quote_mean", "quote_std_sample", "corr_quote_rpm", "corr_quote_market_raw", "corr_quote_market_clean", "proxy_MAE_dollars", "proxy_MAPE_pct"]]),
        "November–December target correlation and proxy errors: NOT AVAILABLE; validation.csv has no posted_rate.",
        "## 4. Non-flagged OOF residuals by month",
        "Absolute-error percentiles: dollars; linear quantile interpolation. Non-flagged: existing retrospective OOF flags.",
        markdown(monthly_nonflag[["month", "n", "mean_signed_pct_error", "MAPE_pct", "abs_error_p90_dollars", "abs_error_p99_dollars"]]),
        "January–June OOF errors: NOT AVAILABLE; no saved held-out predictions for those months.",
        "## File checks",
        markdown(pd.DataFrame([checks])),
        "Added: audit/residual_tail_audit.py; audit/residual_tail/ CSVs, protocols, checks and REPORT.md."]
    (OUT / "REPORT.md").write_text("\n\n".join(report)+"\n", encoding="utf-8")
    print("Wrote audit/residual_tail/REPORT.md; protected files unchanged.", flush=True)


if __name__ == "__main__":
    main()
