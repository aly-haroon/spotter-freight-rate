"""E2: compare fixed 8-, 13-, and 17-week recent-price windows.

Only the prediction-level shift changes with this window. The tree target and
features are identical, so one tree fit per fold is sufficient for all windows.
Submission files and production defaults are unchanged.
"""
import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd

from src.config import FOLDS, REPORTS_DIR, TRAIN_PATH
from src.data import clean, flag_outliers, load
from src.model import MarketLayer, TwoStageModel
from src.predict import check_predictions
from src.validate import metrics

OUTPUT_DIR = REPORTS_DIR / "window_experiment"
WINDOWS = (8, 13, 17)


def main(prepare_only: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = {
        "windows_weeks": WINDOWS,
        "reference_weeks": 13,
        "folds": FOLDS,
        "acceptance": [
            "Filtered diagnostic fold-mean MAPE improves by at least 0.05 percentage points",
            "No filtered diagnostic fold MAPE worsens by more than 0.3 percentage points",
            "All-row fold-mean MAE and MAPE do not worsen",
        ],
        "notes": [
            "Three tree fits total; all windows reuse identical trees within each fold",
            "Window affects prediction shift only, not tree training target or features",
            "All-row results are primary; filtered rows are retrospective diagnostics",
            "Previously used overlapping development folds, not independent final evaluation",
            "No production setting is changed automatically",
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

    rows, frames, timings = [], [], []
    for fold, start, end in FOLDS:
        train = clean(raw.loc[raw["date"] < start])
        train = train.loc[~flag_outliers(train)]
        test = clean(raw.loc[(raw["date"] >= start) & (raw["date"] < end)])
        print(f"{fold}: fitting shared recent-level trees on {len(train):,} rows...", flush=True)
        started = perf_counter()
        model = TwoStageModel().fit(train)
        reference = model.predict(test)
        seconds = perf_counter() - started
        timings.append({"fold": fold, "fit_and_predict_seconds": seconds})
        print(f"Completed in {seconds:.1f} seconds", flush=True)
        normal = ~flag_outliers(test).to_numpy()  # retrospective evaluation only
        segments = {"all rows": np.ones(len(test), dtype=bool), "filtered diagnostic": normal}
        frame = test[["load_id", "date", "posted_rate"]].copy()
        frame["fold"] = fold
        frame["diagnostic_outlier"] = ~normal
        for weeks in WINDOWS:
            layer = MarketLayer(level="recent", recent_weeks=weeks).fit(train)
            # Confirms the window did not change the market training offset.
            np.testing.assert_allclose(layer.training_offset(train), model.market_.training_offset(train), rtol=0, atol=1e-12)
            predicted = reference * np.exp(layer.shift_ - model.market_.shift_)
            check_predictions(predicted, f"{fold} {weeks}-week window")
            frame[f"prediction_{weeks}w"] = predicted
            for segment, mask in segments.items():
                if not mask.any():
                    raise ValueError(f"Empty evaluation segment: {fold} {segment}")
                rows.append({"fold": fold, "weeks": weeks, "segment": segment,
                             "n": int(mask.sum()), "level_shift_log": layer.shift_,
                             **metrics(test["posted_rate"].to_numpy()[mask], predicted[mask])})
        frames.append(frame)

    results = pd.DataFrame(rows)
    summary = results.groupby(["segment", "weeks"])[["MAE", "RMSE", "MAPE%"]].mean()
    filtered = results.loc[results["segment"] == "filtered diagnostic"].pivot(index="fold", columns="weeks", values="MAPE%")
    all_rows = summary.loc["all rows"]
    decisions = []
    for weeks in (8, 17):
        improvement = float((filtered[13] - filtered[weeks]).mean())
        worst = float((filtered[weeks] - filtered[13]).max())
        eligible = bool(improvement >= 0.05 and worst <= 0.3
                        and all_rows.loc[weeks, "MAE"] <= all_rows.loc[13, "MAE"]
                        and all_rows.loc[weeks, "MAPE%"] <= all_rows.loc[13, "MAPE%"])
        decisions.append({"weeks": weeks, "eligible_for_review": eligible,
                          "filtered_mean_mape_improvement_points": improvement,
                          "worst_filtered_fold_mape_worsening_points": worst})
    results.to_csv(OUTPUT_DIR / "results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv")
    pd.concat(frames, ignore_index=True).to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    pd.DataFrame(timings).to_csv(OUTPUT_DIR / "timings.csv", index=False)
    (OUTPUT_DIR / "decision.json").write_text(json.dumps({"candidates": decisions, "production_changed": False}, indent=2) + "\n", encoding="utf-8")
    print("\n=== Fold-mean errors ===\n", summary.round(3).to_string())
    print("\n=== Filtered diagnostic MAPE by fold ===\n", filtered.round(3).to_string())
    print("\nAcceptance recommendations:", json.dumps(decisions, indent=2))
    print(f"Saved experiment outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Check splits without model fitting")
    main(prepare_only=parser.parse_args().prepare_only)
