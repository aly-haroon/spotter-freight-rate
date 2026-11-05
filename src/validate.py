"""Rolling-origin backtest and ablations.

Each fold trains on all months before the test window and predicts the next two months,
mirroring the real task (Jan-Oct -> Nov-Dec). Run:  python -m src.validate
"""
import numpy as np
import pandas as pd

from src.config import FOLDS, REPORTS_DIR, TRAIN_PATH
from src.data import DISTANCE_BANDS, clean, flag_outliers, load
from src.model import TwoStageModel


class BaselineModel:
    """Median rate-per-mile by equipment and distance band, times distance."""

    def fit(self, train):
        self.table_ = train.groupby([train["equipment"], pd.cut(train["distance"], DISTANCE_BANDS)], observed=True)["rpm"].median()
        return self

    def predict(self, df):
        keys = pd.MultiIndex.from_arrays([df["equipment"], pd.cut(df["distance"], DISTANCE_BANDS)])
        return self.table_.reindex(keys).to_numpy() * df["distance"].to_numpy()


# name -> (model factory, drop outliers from training?)
CHOSEN = "5 CHOSEN: market effect + recent level"
VARIANTS = {
    "1 baseline (equipment x distance median)": (BaselineModel, True),
    "2 trees, no market info": (lambda: TwoStageModel(use_market_layer=False), True),
    "3 trees + city/lane premiums": (lambda: TwoStageModel(use_market_layer=False, use_premiums=True), True),
    "4 trees + raw market/time features": (lambda: TwoStageModel(use_market_layer=False, extra_tree_features=["market_index", "t_months"]), True),
    CHOSEN: (TwoStageModel, True),
    "6 market effect, no level update": (lambda: TwoStageModel(level="none"), True),
    "7 market effect + straight trend line": (lambda: TwoStageModel(level="trend"), True),
    "8 chosen + quote_signal": (lambda: TwoStageModel(extra_tree_features=["quote_signal"]), True),
    "9 chosen, outliers kept": (TwoStageModel, False),
}


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    return {
        "MAE": np.abs(error).mean(),
        "RMSE": np.sqrt((error ** 2).mean()),
        "MAPE%": 100 * np.mean(np.abs(error) / actual),
    }


def run_backtest(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, start, end in FOLDS:
        train = data[data["date"] < start]
        test = data[(data["date"] >= start) & (data["date"] < end)]
        train_outlier = flag_outliers(train)
        test_outlier = flag_outliers(test)  # only used to label evaluation rows
        train_lanes = set(zip(train["pickup"], train["delivery"]))
        unseen_lane = ~pd.Series(list(zip(test["pickup"], test["delivery"])), index=test.index).isin(train_lanes)

        for name, (factory, drop_outliers) in VARIANTS.items():
            fit_data = train[~train_outlier] if drop_outliers else train
            pred = factory().fit(fit_data).predict(test)
            actual = test["posted_rate"].to_numpy()
            normal = ~test_outlier.to_numpy()
            segments = {
                "normal rows": normal,
                "all rows": np.ones(len(test), bool),
                "normal, seen lanes": normal & ~unseen_lane.to_numpy(),
                "normal, unseen lanes": normal & unseen_lane.to_numpy(),
            }
            for segment, mask in segments.items():
                rows.append({"fold": fold, "variant": name, "segment": segment, "n": int(mask.sum()),
                             **metrics(actual[mask], pred[mask])})
            print(f"{fold} {name:45s} MAPE {rows[-4]['MAPE%']:.2f}%")
    return pd.DataFrame(rows)


def main() -> None:
    data = clean(load(TRAIN_PATH))
    results = run_backtest(data)
    REPORTS_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORTS_DIR / "backtest_results.csv", index=False)

    pd.set_option("display.width", 200)
    summary = results.groupby(["segment", "variant"])[["MAE", "RMSE", "MAPE%"]].mean().round(2)
    print("\n=== Mean over folds, normal rows ===")
    print(summary.loc["normal rows"])
    print("\n=== Mean over folds, all rows (incl. outliers) ===")
    print(summary.loc["all rows"])
    print("\n=== Chosen model by lane type ===")
    print(summary.xs(CHOSEN, level="variant"))
    print("\n=== Chosen model per fold, normal rows ===")
    chosen = results[(results["variant"] == CHOSEN) & (results["segment"] == "normal rows")]
    print(chosen[["fold", "n", "MAE", "RMSE", "MAPE%"]].round(2).to_string(index=False))
    print("\n=== MAPE% per fold, every variant, normal rows ===")
    normal = results[results["segment"] == "normal rows"]
    print(normal.pivot(index="variant", columns="fold", values="MAPE%").round(2))


if __name__ == "__main__":
    main()
