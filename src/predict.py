"""
Load the saved model + scaler once, then predict RUL for a single engine's
sensor history. This is the same code path the Streamlit app uses, so a
prediction made in the app matches a prediction made from the CLI exactly.
"""

import json

import joblib
import pandas as pd

from config import DROP_COLUMNS, FEATURE_COLUMNS, WINDOW_SIZE
from features import _window_stats

MODELS_DIR = "../models"


class RULPredictor:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.model = joblib.load(f"{models_dir}/xgb_model.joblib")
        self.scaler = joblib.load(f"{models_dir}/scaler.joblib")
        with open(f"{models_dir}/feature_names.json") as f:
            self.feature_names = json.load(f)

    def predict_one(self, engine_df: pd.DataFrame) -> float:
        """
        engine_df: raw-format rows (the 26 C-MAPSS columns, or at least the
        17 retained feature columns + time_cycles) for ONE engine, sorted or
        unsorted, covering at least the last WINDOW_SIZE cycles.

        Returns a single predicted RUL (in cycles) for "right now"
        (i.e. at the engine's most recent cycle in engine_df).
        """
        df = engine_df.copy()
        if "time_cycles" in df.columns:
            df = df.sort_values("time_cycles")
        if len(df) < WINDOW_SIZE:
            raise ValueError(
                f"Need at least {WINDOW_SIZE} cycles of history, got {len(df)}."
            )
        window = df.tail(WINDOW_SIZE)

        stats = _window_stats(window, FEATURE_COLUMNS)
        X = pd.DataFrame([stats])[self.feature_names]
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)[0]
        return max(0.0, float(pred))  # RUL can't be negative


if __name__ == "__main__":
    from data_prep import load_rul, prepare_test

    test_df = prepare_test("../data/test_FD001.txt")
    true_rul = load_rul("../data/RUL_FD001.txt")

    predictor = RULPredictor()

    # Spot-check a few engines
    for unit in [1, 2, 3, 50, 100]:
        engine_df = test_df[test_df["unit_number"] == unit]
        pred = predictor.predict_one(engine_df)
        true = true_rul.loc[unit]
        print(f"engine {unit}: predicted RUL = {pred:.1f}, true RUL = {true}")
