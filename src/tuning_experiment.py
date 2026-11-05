"""E4: final fixed six-setting XGBoost comparison on development time folds.

Run: python -u -m src.tuning_experiment
This experiment never changes production defaults or submission files.
"""
import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd

from src.config import FOLDS, REPORTS_DIR, TRAIN_PATH
from src.data import clean, flag_outliers, load
from src.model import GBM_PARAMS, TwoStageModel
from src.predict import check_predictions
from src.validate import metrics

OUTPUT_DIR = REPORTS_DIR / "tuning_experiment"
REFERENCE = "depth6_lr003"
SETTINGS = [
    {"name": f"depth{depth}_lr{suffix}",
     "params": {**GBM_PARAMS, "max_depth": depth,
                "learning_rate": rate, "n_estimators": trees}}
    for depth in (4, 6, 8)
    for rate, trees, suffix in ((0.03, 1500, "003"), (0.05, 900, "005"))
]


def main(prepare_only: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = {
        "settings": SETTINGS,
        "reference": REFERENCE,
        "folds": FOLDS,
        "acceptance": [
            "Filtered diagnostic fold-mean MAPE improves by at least 0.05 percentage points",
            "No filtered diagnostic fold MAPE worsens by more than 0.3 percentage points",
            "All-row fold-mean MAE and MAPE do not worsen",
        ],
        "notes": [
            "Fixed grid; 1500 trees at 0.03 and 900 at 0.05, with equal rate-times-tree-count",
            "Same features, market layer, 13-week level, seed and cleaning for every setting",
            "All-row metrics are primary; filtered rows are retrospective diagnostics",
            "Previously used overlapping development folds; not independent final accuracy",
            "No early stopping or extra parameter search using these results",
            "No production settings or submission files change automatically",
        ],
    }
    (OUTPUT_DIR / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    raw = load(TRAIN_PATH)
    for fold, start, end in FOLDS:
        train = raw.loc[raw["date"] < start]
        test = raw.loc[(raw["date"] >= start) & (raw["date"] < end)]
        if train.empty or test.empty or train["date"].max() >= test["date"].min():
            raise ValueError(f"Invalid time split: {fold}")
        print(f"{fold}: {len(train):,} training rows, {len(test):,} test rows", flush=True)
    print(f"Fixed grid: {len(SETTINGS)} settings x {len(FOLDS)} folds = {len(SETTINGS) * len(FOLDS)} fits", flush=True)
    if prepare_only:
        print("Split checks passed. No training performed.", flush=True)
        return

    rows, timings = [], []
    for fold, start, end in FOLDS:
        train = clean(raw.loc[raw["date"] < start])
        train = train.loc[~flag_outliers(train)]
        test = clean(raw.loc[(raw["date"] >= start) & (raw["date"] < end)])
        normal = ~flag_outliers(test).to_numpy()  # diagnostic labels only
        segments = {"all rows": np.ones(len(test), dtype=bool), "filtered diagnostic": normal}
        frame = test[["load_id", "date", "posted_rate"]].copy()
        frame["fold"] = fold
        frame["diagnostic_outlier"] = ~normal
        for setting in SETTINGS:
            name = setting["name"]
            params = setting["params"]
            print(f"{fold}: fitting {name}, {len(train):,} rows, {params['n_estimators']} trees...", flush=True)
            started = perf_counter()
            predicted = TwoStageModel(gbm_params=dict(params)).fit(train).predict(test)
            seconds = perf_counter() - started
            check_predictions(predicted, f"{fold} {name}")
            timings.append({"fold": fold, "model": name, "fit_and_predict_seconds": seconds})
            frame[f"prediction_{name}"] = predicted
            for segment, mask in segments.items():
                if not mask.any():
                    raise ValueError(f"Empty evaluation segment: {fold} {segment}")
                rows.append({"fold": fold, "model": name, "segment": segment,
                             "n": int(mask.sum()), **metrics(test["posted_rate"].to_numpy()[mask], predicted[mask])})
            # Keep completed metrics and timings if the long run is interrupted.
            pd.DataFrame(rows).to_csv(OUTPUT_DIR / "results.csv", index=False)
            pd.DataFrame(timings).to_csv(OUTPUT_DIR / "timings.csv", index=False)
            frame.to_csv(OUTPUT_DIR / f"predictions_{fold}.csv", index=False)
            print(f"Completed in {seconds:.1f} seconds", flush=True)

    results = pd.DataFrame(rows)
    summary = results.groupby(["segment", "model"])[["MAE", "RMSE", "MAPE%"]].mean()
    filtered = results.loc[results["segment"] == "filtered diagnostic"].pivot(index="fold", columns="model", values="MAPE%")
    all_rows = summary.loc["all rows"]
    decisions = []
    for setting in SETTINGS:
        name = setting["name"]
        if name == REFERENCE:
            continue
        improvement = float((filtered[REFERENCE] - filtered[name]).mean())
        worst = float((filtered[name] - filtered[REFERENCE]).max())
        eligible = bool(improvement >= 0.05 and worst <= 0.3
                        and all_rows.loc[name, "MAE"] <= all_rows.loc[REFERENCE, "MAE"]
                        and all_rows.loc[name, "MAPE%"] <= all_rows.loc[REFERENCE, "MAPE%"])
        decisions.append({"model": name, "eligible_for_review": eligible,
                          "filtered_mean_mape_improvement_points": improvement,
                          "worst_filtered_fold_mape_worsening_points": worst,
                          "all_row_mean_mae": float(all_rows.loc[name, "MAE"]),
                          "all_row_mean_mape": float(all_rows.loc[name, "MAPE%"])})
    # Among eligible candidates, review lowest all-row mean MAPE, then MAE.
    eligible = sorted((d for d in decisions if d["eligible_for_review"]),
                      key=lambda d: (d["all_row_mean_mape"], d["all_row_mean_mae"], d["model"]))
    decision = {"reference": REFERENCE, "candidates": decisions,
                "candidate_for_review": eligible[0]["model"] if eligible else None,
                "production_changed": False}
    summary.to_csv(OUTPUT_DIR / "summary.csv")
    (OUTPUT_DIR / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print("\n=== Fold-mean errors ===\n", summary.round(3).to_string())
    print("\n=== Filtered diagnostic MAPE by fold ===\n", filtered.round(3).to_string())
    print("\nAcceptance recommendations:\n", json.dumps(decision, indent=2))
    print(f"Saved experiment outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Check splits without model fitting")
    main(prepare_only=parser.parse_args().prepare_only)
