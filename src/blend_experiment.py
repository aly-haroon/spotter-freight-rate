"""E3: fixed 50:50 blend of recent-level and trend models across existing folds.

Run: python -u -m src.blend_experiment
No submission files or production model settings are changed.
"""
import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd

from src.config import FOLDS, REPORTS_DIR, TRAIN_PATH
from src.data import clean, flag_outliers, load
from src.model import TwoStageModel
from src.predict import check_predictions
from src.validate import metrics

OUTPUT_DIR = REPORTS_DIR / "blend_experiment"


def main(prepare_only: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = {
        "blend": "exp(0.5 * log(recent_prediction) + 0.5 * log(trend_prediction))",
        "folds": FOLDS,
        "acceptance": [
            "Filtered diagnostic fold-mean MAPE improves by at least 0.05 percentage points",
            "No filtered diagnostic fold MAPE worsens by more than 0.3 percentage points",
            "All-row fold-mean MAE and MAPE do not worsen",
        ],
        "notes": [
            "Fixed weight chosen before this experiment; no weight search",
            "All-row results are primary; filtered rows are retrospective diagnostics",
            "Folds overlap and were previously used for development; not an independent final test",
            "RMSE is reported separately because the hidden evaluation metric is unknown",
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
    if prepare_only:
        print("Split checks passed. No training performed.", flush=True)
        return

    rows, prediction_frames, timings = [], [], []
    for fold, start, end in FOLDS:
        train = clean(raw.loc[raw["date"] < start])
        train = train.loc[~flag_outliers(train)]
        test = clean(raw.loc[(raw["date"] >= start) & (raw["date"] < end)])
        predictions = {}
        for level in ["recent", "trend"]:
            model = TwoStageModel(level=level)
            print(f"{fold}: fitting {level}, {len(train):,} rows, {model.gbm_params['n_estimators']} trees...", flush=True)
            started = perf_counter()
            predictions[level] = model.fit(train).predict(test)
            seconds = perf_counter() - started
            check_predictions(predictions[level], f"{fold} {level}")
            timings.append({"fold": fold, "model": level, "fit_and_predict_seconds": seconds})
            print(f"Completed in {seconds:.1f} seconds", flush=True)
        predictions["blend_50_50"] = np.exp(0.5 * np.log(predictions["recent"]) + 0.5 * np.log(predictions["trend"]))
        check_predictions(predictions["blend_50_50"], f"{fold} blend")
        normal = ~flag_outliers(test).to_numpy()  # evaluation subset only
        segments = {"all rows": np.ones(len(test), dtype=bool), "filtered diagnostic": normal}
        frame = test[["load_id", "date", "posted_rate"]].copy()
        frame["fold"] = fold
        frame["diagnostic_outlier"] = ~normal
        for name, predicted in predictions.items():
            frame[f"prediction_{name}"] = predicted
            for segment, mask in segments.items():
                if not mask.any():
                    raise ValueError(f"Empty evaluation segment: {fold} {segment}")
                rows.append({"fold": fold, "model": name, "segment": segment, "n": int(mask.sum()),
                             **metrics(test["posted_rate"].to_numpy()[mask], predicted[mask])})
        prediction_frames.append(frame)

    results = pd.DataFrame(rows)
    summary = results.groupby(["segment", "model"])[["MAE", "RMSE", "MAPE%"]].mean()
    filtered = results.loc[results["segment"] == "filtered diagnostic"].pivot(index="fold", columns="model", values="MAPE%")
    improvement = float((filtered["recent"] - filtered["blend_50_50"]).mean())
    worst_worsening = float((filtered["blend_50_50"] - filtered["recent"]).max())
    all_rows = summary.loc["all rows"]
    eligible = bool(improvement >= 0.05 and worst_worsening <= 0.3
                    and all_rows.loc["blend_50_50", "MAE"] <= all_rows.loc["recent", "MAE"]
                    and all_rows.loc["blend_50_50", "MAPE%"] <= all_rows.loc["recent", "MAPE%"])
    decision = {"eligible_for_review": eligible,
                "filtered_mean_mape_improvement_points": improvement,
                "worst_filtered_fold_mape_worsening_points": worst_worsening,
                "production_changed": False}
    results.to_csv(OUTPUT_DIR / "results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv")
    pd.concat(prediction_frames, ignore_index=True).to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    pd.DataFrame(timings).to_csv(OUTPUT_DIR / "timings.csv", index=False)
    (OUTPUT_DIR / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print("\n=== Fold-mean errors ===\n", summary.round(3).to_string())
    print("\n=== Filtered diagnostic MAPE by fold ===\n", filtered.round(3).to_string())
    print(f"\nMeets experiment acceptance rules: {eligible}. Review results before adoption.")
    print(f"Saved experiment outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Check splits without model fitting")
    main(prepare_only=parser.parse_args().prepare_only)
