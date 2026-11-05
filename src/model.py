"""Two-stage freight rate model.

    log(rpm) = trees(distance, equipment, weight, location)   # stage 2: gradient boosting
             + b * log(market_index)                           # stage 1: market effect
             + level_shift                                     # stage 1: recent price level
    rate     = exp(log(rpm)) * distance
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import RECENT_LEVEL_WEEKS, SEED
from src.features import PremiumEncoder, base_features, out_of_fold_premiums

GBM_PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,
)

LANE = ["pickup", "delivery", "equipment"]


def _demean(df: pd.DataFrame, cols: list[str], group_sets: list[list[str]], sweeps: int = 5) -> pd.DataFrame:
    """Remove group means for several groupings at once (alternating projections)."""
    out = df[cols].copy()
    for _ in range(sweeps):
        for keys in group_sets:
            out = out - out.groupby([df[k] for k in keys]).transform("mean")
    return out


def _slope(x: pd.Series, y: pd.Series) -> float:
    return float((x * y).sum() / (x ** 2).sum())


@dataclass
class MarketLayer:
    """Price level over time: market_index effect plus where prices have recently been.

    level:
      "recent" - level = average of the last RECENT_LEVEL_WEEKS weeks (after removing the
                 market effect). Robust: the slow drift is real but unstable to extrapolate.
      "trend"  - straight-line trend over the whole training window, extrapolated.
      "none"   - market effect only; level = whole-training average.
    """

    level: str = "recent"
    recent_weeks: int = RECENT_LEVEL_WEEKS

    def fit(self, df: pd.DataFrame) -> "MarketLayer":
        df = df.assign(log_mi=np.log(df["market_index"]), week=df["date"].dt.to_period("W").astype(str))

        # Market effect from same-lane loads in the same week: time barely moves inside a
        # week, so the slow drift cannot be mistaken for a market effect.
        within_week = _demean(df, ["log_rpm", "log_mi"], [LANE, ["week"]])
        self.b_market_ = _slope(within_week["log_mi"], within_week["log_rpm"])

        # De-marketed price level of each load, with lane differences removed.
        adjusted = df["log_rpm"] - self.b_market_ * df["log_mi"]
        level = adjusted - adjusted.groupby([df[k] for k in LANE]).transform("mean")

        self.trend_ = 0.0
        self.shift_ = 0.0
        if self.level == "recent":
            if self.recent_weeks <= 0:
                raise ValueError("recent_weeks must be positive")
            recent = df["date"] > df["date"].max() - pd.Timedelta(weeks=self.recent_weeks)
            self.shift_ = float(level[recent].mean())
        elif self.level == "trend":
            within_lane = _demean(df.assign(adjusted=adjusted), ["adjusted", "t_months"], [LANE])
            self.trend_ = _slope(within_lane["t_months"], within_lane["adjusted"])
            self.t_center_ = float(df["t_months"].mean())
        return self

    def training_offset(self, df: pd.DataFrame) -> pd.Series:
        """Market effect removed from the training target (the trees learn the average level)."""
        offset = self.b_market_ * np.log(df["market_index"])
        if self.level == "trend":
            offset = offset + self.trend_ * (df["t_months"] - self.t_center_)
        return offset

    def prediction_offset(self, df: pd.DataFrame) -> pd.Series:
        return self.training_offset(df) + self.shift_


@dataclass
class TwoStageModel:
    use_market_layer: bool = True   # stage 1 on/off
    level: str = "recent"           # see MarketLayer
    use_premiums: bool = False      # city/lane target encodings (did not help in the backtest)
    extra_tree_features: list[str] = field(default_factory=list)  # e.g. ["quote_signal"] for ablations
    gbm_params: dict = field(default_factory=lambda: dict(GBM_PARAMS))

    def _tree_input(self, df: pd.DataFrame, premiums: pd.DataFrame | None) -> pd.DataFrame:
        X = base_features(df)
        for col in self.extra_tree_features:
            X[col] = df[col]
        if premiums is not None:
            X = X.join(premiums)
        return X

    def fit(self, train: pd.DataFrame) -> "TwoStageModel":
        target = train["log_rpm"]
        if self.use_market_layer:
            self.market_ = MarketLayer(self.level).fit(train)
            target = target - self.market_.training_offset(train)

        premiums = None
        if self.use_premiums:
            premiums = out_of_fold_premiums(train, target)
            self.encoder_ = PremiumEncoder().fit(train, target)

        self.trees_ = xgb.XGBRegressor(**self.gbm_params)
        self.trees_.fit(self._tree_input(train, premiums), target)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        premiums = self.encoder_.transform(df) if self.use_premiums else None
        log_rpm = self.trees_.predict(self._tree_input(df, premiums))
        if self.use_market_layer:
            log_rpm = log_rpm + self.market_.prediction_offset(df).to_numpy()
        return np.exp(log_rpm) * df["distance"].to_numpy()
