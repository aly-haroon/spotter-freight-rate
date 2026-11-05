"""Feature engineering, including leakage-safe city/lane premiums (target encodings)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.config import SEED
from src.data import DISTANCE_BANDS

EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]

# Grouping keys for premiums, with how strongly each is pulled toward zero
# (smoothing = number of "virtual" average loads added to every group).
ENCODINGS = {
    "pickup": (["pickup"], 50),
    "delivery": (["delivery"], 50),
    "lane": (["pickup", "delivery"], 20),
    "lane_equipment": (["pickup", "delivery", "equipment"], 20),
}


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features computed from a load's own columns (no target involved)."""
    out = pd.DataFrame(index=df.index)
    out["log_distance"] = np.log(df["distance"])
    for name in EQUIPMENT:
        out[f"eq_{name}"] = (df["equipment"] == name).astype(int)
    out["weight"] = df["weight"]
    out["weight_missing"] = df["weight_missing"]
    # Coordinates are shifted/anonymised but consistent per city, so they still
    # tell the model where an unseen city sits relative to known ones.
    for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        out[col] = df[col]
    out["delta_lat"] = df["delivery_lat"] - df["pickup_lat"]
    out["delta_lon"] = df["delivery_lon"] - df["pickup_lon"]
    return out


def _keys(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).agg("|".join, axis=1)


def _distance_band(df: pd.DataFrame) -> pd.Series:
    return pd.cut(df["distance"], DISTANCE_BANDS).astype(str)


class PremiumEncoder:
    """Smoothed average residual per city / lane, relative to similar loads.

    residual = target - mean target of loads with the same equipment and distance band,
    so a premium says "how much pricier is this place than a typical comparable load".
    Unseen keys get 0 (assume average).
    """

    def fit(self, df: pd.DataFrame, target: pd.Series) -> "PremiumEncoder":
        self.baseline_ = target.groupby([df["equipment"], _distance_band(df)]).mean()
        residual = target - self._baseline(df)
        self.tables_ = {}
        for name, (cols, smoothing) in ENCODINGS.items():
            grouped = residual.groupby(_keys(df, cols)).agg(["sum", "count"])
            self.tables_[name] = (grouped["sum"] / (grouped["count"] + smoothing), grouped["count"])
        return self

    def _baseline(self, df: pd.DataFrame) -> pd.Series:
        index = pd.MultiIndex.from_arrays([df["equipment"], _distance_band(df)])
        values = self.baseline_.reindex(index).to_numpy()
        return pd.Series(np.nan_to_num(values, nan=float(np.nanmean(self.baseline_))), index=df.index)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for name, (cols, _) in ENCODINGS.items():
            premium, count = self.tables_[name]
            keys = _keys(df, cols)
            out[f"prem_{name}"] = keys.map(premium).fillna(0.0).to_numpy()
            out[f"count_{name}"] = keys.map(count).fillna(0).to_numpy()
        return out


def out_of_fold_premiums(df: pd.DataFrame, target: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """Premiums for training rows, each computed without the row's own price."""
    parts = []
    for fit_idx, apply_idx in KFold(n_splits, shuffle=True, random_state=SEED).split(df):
        encoder = PremiumEncoder().fit(df.iloc[fit_idx], target.iloc[fit_idx])
        parts.append(encoder.transform(df.iloc[apply_idx]))
    return pd.concat(parts).loc[df.index]
