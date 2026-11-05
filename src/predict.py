"""Final training on all labelled data and creation of both submission files.

Run:  python -m src.predict
Writes:
  validation_predictions.csv          load_id,predicted_rate for all 12,000 validation loads
  data/december_chart_inputs.csv      the 31 December rows with predicted_rate filled in
"""
import numpy as np
import pandas as pd

from src.config import DECEMBER_PATH, PREDICTIONS_PATH, TEMPLATE_PATH, TRAIN_PATH, VALIDATION_PATH
from src.data import clean, flag_outliers, load
from src.model import TwoStageModel


def city_coordinates(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """One (lat, lon) per city, taken from both pickup and delivery columns."""
    parts = []
    for df in frames:
        for side in ["pickup", "delivery"]:
            parts.append(df[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city")


def build_december_features(december: pd.DataFrame, validation: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    """The chart file has no coordinates or market_index, so fill them in.

    market_index for each date = mean over the validation loads on that date (features only,
    no labels), i.e. the market level the model will also see for real Nov-Dec loads.
    """
    df = december.drop(columns="predicted_rate").copy()
    df["date"] = pd.to_datetime(df["date"])
    for side in ["pickup", "delivery"]:
        coords = cities.reindex(df[side])
        if coords.isna().any().any():
            raise SystemExit(f"Unknown {side} city in December inputs: {df[side].unique()}")
        df[f"{side}_lat"] = coords["lat"].to_numpy()
        df[f"{side}_lon"] = coords["lon"].to_numpy()
    daily_mi = validation.groupby("date")["market_index"].mean()
    df["market_index"] = df["date"].map(daily_mi)
    if df["market_index"].isna().any():
        raise SystemExit("A December date has no validation loads to take market_index from")
    df["quote_signal"] = np.nan  # not used by the model
    return clean(df)


def check_predictions(values: np.ndarray, label: str) -> None:
    if not np.isfinite(values).all() or (values <= 0).any():
        raise SystemExit(f"{label}: predictions must be finite and positive")


def main() -> None:
    train = clean(load(TRAIN_PATH))
    train = train[~flag_outliers(train)]
    validation = clean(load(VALIDATION_PATH))
    print(f"Training on {len(train):,} loads ({train['date'].min():%Y-%m-%d} to {train['date'].max():%Y-%m-%d})")

    model = TwoStageModel().fit(train)
    print(f"Market effect b = {model.market_.b_market_:.3f}, recent level shift = {100 * model.market_.shift_:+.2f}%")

    # Validation predictions, in the template's row order.
    template = pd.read_csv(TEMPLATE_PATH)
    predicted = pd.Series(model.predict(validation), index=validation["load_id"])
    template["predicted_rate"] = predicted.reindex(template["load_id"]).round(2).to_numpy()
    check_predictions(template["predicted_rate"].to_numpy(), "validation")
    template[["load_id", "predicted_rate"]].to_csv(PREDICTIONS_PATH, index=False)
    print(f"Wrote {len(template):,} predictions to {PREDICTIONS_PATH.name}")

    # December chart: original 7 columns, only predicted_rate filled.
    december = pd.read_csv(DECEMBER_PATH)
    features = build_december_features(december, validation, city_coordinates([train, validation]))
    december["predicted_rate"] = model.predict(features).round(2)
    check_predictions(december["predicted_rate"].to_numpy(), "December")
    december.to_csv(DECEMBER_PATH, index=False)
    print(f"Wrote December chart inputs: {december['predicted_rate'].min():.0f} to {december['predicted_rate'].max():.0f} USD")


if __name__ == "__main__":
    main()
