"""Project-wide paths and constants."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"

SEED = 42

# Time origin for the trend feature (months since this date).
START_DATE = pd.Timestamp("2025-01-01")

# A training load is an outlier if its rate-per-mile is outside this band
# relative to a robust reference price for comparable loads.
OUTLIER_BAND = (0.6, 1.6)

# The forecast price level is the (market-adjusted) average of this many final training weeks.
RECENT_LEVEL_WEEKS = 13

# Rolling-origin backtest: train on everything before `start`, test on [start, end).
FOLDS = [
    ("F1", "2025-07-01", "2025-09-01"),
    ("F2", "2025-08-01", "2025-10-01"),
    ("F3", "2025-09-01", "2025-11-01"),
]
