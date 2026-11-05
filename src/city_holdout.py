"""E1: hide eight training cities and evaluate the fixed September-October window.

Run: python -u -m src.city_holdout
Use --prepare-only to inspect the split without training.
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR, SEED, TRAIN_PATH
from src.data import clean, flag_outliers, load
from src.model import TwoStageModel
from src.predict import check_predictions
from src.validate import metrics

TEST_START = pd.Timestamp("2025-09-01")
TEST_END = pd.Timestamp("2025-11-01")
HELDOUT_CITY_COUNT = 8
OUTPUT_DIR = REPORTS_DIR / "city_holdout"


def touches_cities(frame: pd.DataFrame, cities: list[str]) -> pd.Series:
    return frame["pickup"].isin(cities) | frame["delivery"].isin(cities)


def prepare_split(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Choose cities from past features only; test prices do not affect selection."""
    past = raw.loc[raw["date"] < TEST_START].copy()
    test = raw.loc[(raw["date"] >= TEST_START) & (raw["date"] < TEST_END)].copy()
    cities = sorted(set(past["pickup"]) | set(past["delivery"]))
    if len(cities) <= HELDOUT_CITY_COUNT or test.empty:
        raise ValueError("Insufficient cities or test rows for the prescribed split")
    heldout = sorted(np.random.default_rng(SEED).choice(cities, HELDOUT_CITY_COUNT, replace=False).tolist())
    restricted = past.loc[~touches_cities(past, heldout)].copy()
    if restricted.empty or touches_cities(restricted, heldout).any():
        raise ValueError("Invalid city exclusion")
    if restricted["date"].max() >= test["date"].min():
        raise ValueError("Training and test dates overlap")
    if not raw["load_id"].is_unique:
        raise ValueError("Duplicate load IDs")
    if not touches_cities(test, heldout).any():
        raise ValueError("No test shipments touch held-out cities")
    return past, restricted, test, heldout


def run(prepare_only: bool = False) -> None:
    past, restricted, raw_test, heldout = prepare_split(load(TRAIN_PATH))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": SEED,
        "heldout_cities": heldout,
        "selection": "Eight cities sampled from sorted January-August city names; no test labels used",
        "training_end_exclusive": str(TEST_START.date()),
        "test_end_exclusive": str(TEST_END.date()),
        "past_rows": len(past),
        "past_rows_after_city_exclusion": len(restricted),
        "test_rows": len(raw_test),
        "test_rows_touching_heldout_cities": int(touches_cities(raw_test, heldout).sum()),
        "notes": [
            "One predetermined city split is a diagnostic, not a guarantee for every new city.",
            "All-row metrics are primary; filtered metrics use test prices for diagnostic labels only.",
            "The reference model sees these cities in past data; both models predict identical test rows.",
            "City removal also reduces training size; the comparison includes that effect.",
            "Same-day market imputation uses feature-only observations from the respective input batch.",
        ],
    }
    # Save the predetermined selection before any model fitting.
    (OUTPUT_DIR / "split.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Held-out cities:", ", ".join(heldout), flush=True)
    print(f"Past rows: {len(past):,}; after hiding cities: {len(restricted):,}", flush=True)
    print(f"Test rows: {len(raw_test):,}; touching hidden cities: {manifest['test_rows_touching_heldout_cities']:,}", flush=True)
    if prepare_only:
        print("Split checks passed. No models trained.", flush=True)
        return

    test = clean(raw_test)
    hidden = touches_cities(test, heldout).to_numpy()
    # Retrospective diagnostic only: never used in training or prediction.
    normal = ~flag_outliers(test).to_numpy()
    segments = {
        "all rows": np.ones(len(test), dtype=bool),
        "all rows, hidden cities": hidden,
        "all rows, other cities": ~hidden,
        "filtered diagnostic": normal,
        "filtered diagnostic, hidden cities": normal & hidden,
        "filtered diagnostic, other cities": normal & ~hidden,
    }
    rows = []
    predictions = test[["load_id", "date", "pickup", "delivery", "posted_rate"]].copy()
    predictions["touches_hidden_city"] = hidden
    predictions["diagnostic_outlier"] = ~normal
    for name, train_raw in [("reference", past), ("cities_hidden", restricted)]:
        train = clean(train_raw)
        train = train.loc[~flag_outliers(train)]
        print(f"Training {name} on {len(train):,} rows...", flush=True)
        predicted = TwoStageModel().fit(train).predict(test)
        check_predictions(predicted, name)
        predictions[f"prediction_{name}"] = predicted
        actual = test["posted_rate"].to_numpy()
        for segment, mask in segments.items():
            if not mask.any():
                raise ValueError(f"Empty evaluation segment: {segment}")
            rows.append({"model": name, "segment": segment, "n": int(mask.sum()),
                         "training_rows": len(train), **metrics(actual[mask], predicted[mask])})
        print(f"Completed {name}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "results.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    reference = results.loc[results["model"] == "reference"].set_index("segment")
    hidden_results = results.loc[results["model"] == "cities_hidden"].set_index("segment")
    delta = hidden_results[["MAE", "RMSE", "MAPE%"]] - reference[["MAE", "RMSE", "MAPE%"]]
    delta.to_csv(OUTPUT_DIR / "error_change.csv")
    print("\n=== All-row errors (primary) ===")
    print(results.loc[results["segment"].str.startswith("all rows")].round(3).to_string(index=False))
    print("\n=== Filtered errors (retrospective diagnostics) ===")
    print(results.loc[results["segment"].str.startswith("filtered")].round(3).to_string(index=False))
    print("\n=== Error change after hiding cities (positive = worse) ===")
    print(delta.round(3).to_string())
    print(f"\nSaved results, predictions, and split details to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Check the split without training")
    run(prepare_only=parser.parse_args().prepare_only)
