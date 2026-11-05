"""Loading, cleaning and outlier flagging."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import OUTLIER_BAND, START_DATE

DISTANCE_BANDS = [0, 150, 250, 350, 500, 700, 1000, 1300, 1600, 2000, 2500, 5000]


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data-quality issues. Uses only the frame's own feature columns (never the target)."""
    df = df.copy()

    # Negative weights mirror the positive range exactly (e.g. -47,500 vs the 47,500 cap): sign flips.
    df["weight_was_negative"] = (df["weight"] < 0).astype(int)
    df["weight"] = df["weight"].abs()
    # Missing weights stay NaN; the tree model routes them natively.
    df["weight_missing"] = df["weight"].isna().astype(int)

    # market_index is mostly a daily market level, so a missing value is best
    # filled with the mean of the other loads on the same date.
    df["market_index_missing"] = df["market_index"].isna().astype(int)
    daily_mean = df.groupby("date")["market_index"].transform("mean")
    df["market_index"] = df["market_index"].fillna(daily_mean)

    df["t_months"] = (df["date"] - START_DATE).dt.days / 30.44

    if "posted_rate" in df.columns:
        df["rpm"] = df["posted_rate"] / df["distance"]
        df["log_rpm"] = np.log(df["rpm"])
    return df


def flag_outliers(train: pd.DataFrame) -> pd.Series:
    """True for loads whose rate-per-mile is far from comparable loads.

    Reference = median rpm of loads with the same equipment, distance band and month.
    Must be called on training data only, so test months never influence the rule.
    """
    band = pd.cut(train["distance"], DISTANCE_BANDS)
    month = train["date"].dt.month
    reference = train.groupby([train["equipment"], band, month], observed=True)["rpm"].transform("median")
    ratio = train["rpm"] / reference
    low, high = OUTLIER_BAND
    return ~ratio.between(low, high)
