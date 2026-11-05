"""Reproduces every data finding used in the report. Cheap: no model training.

Run:  python -m src.eda      (tables are printed and saved to reports/eda_*.csv)
"""
import numpy as np
import pandas as pd

from src.config import FOLDS, REPORTS_DIR, TRAIN_PATH, VALIDATION_PATH
from src.data import DISTANCE_BANDS, clean, flag_outliers, load
from src.model import LANE, MarketLayer, _demean


def save(table: pd.DataFrame, name: str, title: str) -> None:
    print(f"\n=== {title} ===")
    print(table.to_string())
    table.to_csv(REPORTS_DIR / f"eda_{name}.csv")


def data_quality(raw_train: pd.DataFrame, raw_val: pd.DataFrame, train: pd.DataFrame) -> None:
    rows = {}
    for label, df in [("train", raw_train), ("validation", raw_val)]:
        rows[label] = {
            "rows": len(df),
            "negative weight": int((df["weight"] < 0).sum()),
            "missing weight": int(df["weight"].isna().sum()),
            "missing market_index": int(df["market_index"].isna().sum()),
            "weight at 47,500 cap": int((df["weight"] == 47_500).sum()),
        }
    rows["train"]["price outliers flagged"] = int(flag_outliers(train).sum())
    train_cities = set(raw_train["pickup"]) | set(raw_train["delivery"])
    val_cities = set(raw_val["pickup"]) | set(raw_val["delivery"])
    train_lanes = set(zip(raw_train["pickup"], raw_train["delivery"]))
    rows["validation"]["cities not in train"] = len(val_cities - train_cities)
    rows["validation"]["rows on unseen lanes"] = int(sum(lane not in train_lanes for lane in zip(raw_val["pickup"], raw_val["delivery"])))
    save(pd.DataFrame(rows).fillna(""), "data_quality", "Data quality")
    print("New cities in validation:", sorted(val_cities - train_cities))


def rpm_structure(train: pd.DataFrame) -> None:
    band = pd.cut(train["distance"], DISTANCE_BANDS)
    table = train.pivot_table(index=band, columns="equipment", values="rpm", aggfunc="median", observed=True).round(3)
    save(table, "rpm_by_distance_equipment", "Median rate-per-mile by distance band and equipment")


def quote_signal_regimes(train: pd.DataFrame, val: pd.DataFrame) -> None:
    """corr(quote_signal, true rpm) per month, and a label-free version that also works on validation."""
    band = pd.cut(train["distance"], DISTANCE_BANDS)
    expected = train.groupby([train["equipment"], band], observed=True)["rpm"].median()

    def expected_rpm(df):
        keys = pd.MultiIndex.from_arrays([df["equipment"], pd.cut(df["distance"], DISTANCE_BANDS)])
        return pd.Series(expected.reindex(keys).to_numpy(), index=df.index)

    def by_month(df, a, b):
        return df.groupby(df["date"].dt.to_period("M")).apply(lambda s: s[a].corr(s[b]), include_groups=False)

    train = train.assign(expected_rpm=expected_rpm(train))
    val = val.assign(expected_rpm=expected_rpm(val))
    table = pd.DataFrame({
        "corr(qs, true rpm)": by_month(train, "quote_signal", "rpm"),
        "corr(qs, expected rpm) label-free": pd.concat([by_month(train, "quote_signal", "expected_rpm"),
                                                         by_month(val, "quote_signal", "expected_rpm")]),
    }).round(3)
    table["regime"] = np.select(
        [table.iloc[:, 1] > 0.5, table.iloc[:, 1] < -0.5], ["copies rpm (leak)", "inverted"], "noise")
    save(table, "quote_signal_regimes", "quote_signal regimes by month (Nov-Dec = validation, label-free)")


def market_effects(train: pd.DataFrame, val: pd.DataFrame) -> None:
    # Weekly cycle of the market index.
    dow = pd.DataFrame({
        "train mean market_index": train.groupby(train["date"].dt.dayofweek)["market_index"].mean(),
        "validation mean market_index": val.groupby(val["date"].dt.dayofweek)["market_index"].mean(),
        "train median rpm": train.groupby(train["date"].dt.dayofweek)["rpm"].median(),
    }).round(3)
    dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    save(dow, "market_weekly_cycle", "Weekly cycle: market_index peaks mid-week, prices follow")

    # Why fitting market + time jointly fails on short windows, and the two-step fix.
    rows = []
    for fold, start, _ in FOLDS + [("all Jan-Oct", "2025-11-01", None)]:
        part = train[train["date"] < start].assign(log_mi=lambda d: np.log(d["market_index"]))
        within = _demean(part, ["log_rpm", "log_mi", "t_months"], [LANE])
        joint, *_ = np.linalg.lstsq(within[["log_mi", "t_months"]].to_numpy(), within["log_rpm"].to_numpy(), rcond=None)
        layer = MarketLayer("recent").fit(part)
        rows.append({
            "train window": f"to {start}",
            "corr(market, time)": round(np.corrcoef(part["log_mi"], part["t_months"])[0, 1], 2),
            "joint fit: market b": round(joint[0], 3),
            "joint fit: trend %/month": round(100 * joint[1], 2),
            "two-step: market b (within-week)": round(layer.b_market_, 3),
            "recent 13-week level shift %": round(100 * layer.shift_, 2),
        })
    save(pd.DataFrame(rows).set_index("train window"), "market_layer_stability",
         "Market effect: joint fit is unstable, within-week estimate is stable")

    # Month-level price level after removing lane and market effects.
    layer = MarketLayer("none").fit(train)
    adjusted = train["log_rpm"] - layer.b_market_ * np.log(train["market_index"])
    level = adjusted - adjusted.groupby([train[k] for k in LANE]).transform("mean")
    monthly = (100 * level.groupby(train["date"].dt.to_period("M")).mean()).round(2).to_frame("de-marketed level %")
    monthly["mean market_index"] = train.groupby(train["date"].dt.to_period("M"))["market_index"].mean().round(3)
    save(monthly, "monthly_level", "Monthly price level after removing lane + market effects")


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    raw_train, raw_val = load(TRAIN_PATH), load(VALIDATION_PATH)
    train_all, val = clean(raw_train), clean(raw_val)
    train = train_all[~flag_outliers(train_all)]

    data_quality(raw_train, raw_val, train_all)
    rpm_structure(train)
    quote_signal_regimes(train, val)
    market_effects(train, val)


if __name__ == "__main__":
    main()
