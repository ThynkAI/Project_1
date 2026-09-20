"""
End-to-end training run.

    python train.py

Loads the data, builds windows, trains four models in increasing order of
sophistication (mean baseline -> Ridge -> Random Forest -> XGBoost), scores
each on a held-out set of engines AND on the official test engines, and
saves the best model (XGBoost) + its scaler + metrics to ../models/.
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import RANDOM_STATE, TRAIN_UNITS, VAL_UNITS, WINDOW_SIZE
from data_prep import load_rul, prepare_test, prepare_train
from features import build_last_window_per_engine, build_training_windows, stat_feature_names
from metrics import evaluate

DATA_DIR = "../data"
MODELS_DIR = "../models"


def main():
    t0 = time.time()

    # ---------- 1. Load + label ----------
    print("Loading data...")
    train_df = prepare_train(f"{DATA_DIR}/train_FD001.txt")
    test_df = prepare_test(f"{DATA_DIR}/test_FD001.txt")
    true_rul = load_rul(f"{DATA_DIR}/RUL_FD001.txt")

    # ---------- 2. Build windows ----------
    # Train/val: EVERY window from every cycle >= WINDOW_SIZE, so the model sees
    # engines at all stages of degradation, not just at the end of life.
    print("Building training windows (engines 1-80)...")
    train_windows = build_training_windows(train_df, unit_ids=TRAIN_UNITS)
    print("Building validation windows (engines 81-100)...")
    val_windows = build_training_windows(train_df, unit_ids=VAL_UNITS)

    # Official test scoring: only the LAST window per test engine (matches how
    # the test trajectories are actually truncated / how the benchmark is scored).
    print("Building official test windows (last 30 cycles per test engine)...")
    test_windows = build_last_window_per_engine(test_df)

    feature_names = stat_feature_names()
    X_train, y_train = train_windows[feature_names], train_windows["RUL"]
    X_val, y_val = val_windows[feature_names], val_windows["RUL"]
    X_test = test_windows[feature_names]
    y_test = true_rul.loc[test_windows["unit_number"]].to_numpy()

    print(f"  train windows: {X_train.shape}, val windows: {X_val.shape}, test engines: {X_test.shape}")

    # ---------- 3. Scale ----------
    # Fit ONLY on training windows. Val and test go through the same
    # already-fitted scaler so nothing about their distribution leaks into it.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = {}

    # ---------- 4a. Baseline: always predict the mean training RUL ----------
    mean_rul = float(y_train.mean())
    val_pred = np.full(len(y_val), mean_rul)
    test_pred = np.full(len(y_test), mean_rul)
    results["baseline_mean"] = {
        "val": evaluate(y_val, val_pred),
        "test": evaluate(y_test, test_pred),
    }

    # ---------- 4b. Ridge regression ----------
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(X_train_s, y_train)
    results["ridge"] = {
        "val": evaluate(y_val, ridge.predict(X_val_s)),
        "test": evaluate(y_test, ridge.predict(X_test_s)),
    }

    # ---------- 4c. Random Forest ----------
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)
    results["random_forest"] = {
        "val": evaluate(y_val, rf.predict(X_val_s)),
        "test": evaluate(y_test, rf.predict(X_test_s)),
    }

    # ---------- 4d. XGBoost (headline model) ----------
    xgb = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(X_train_s, y_train)
    xgb_test_pred = xgb.predict(X_test_s)
    results["xgboost"] = {
        "val": evaluate(y_val, xgb.predict(X_val_s)),
        "test": evaluate(y_test, xgb_test_pred),
    }

    # ---------- 4e. Save per-engine test predictions ----------
    # This is the raw (true, predicted) pair for each of the 100 official test
    # engines -- used by the app's "Model Performance" tab to draw the
    # predicted-vs-true scatter plot. Saved once here so the app doesn't have
    # to re-run the model over the whole test set just to display it.
    test_predictions = pd.DataFrame(
        {
            "unit_number": test_windows["unit_number"].values,
            "true_rul": y_test,
            "predicted_rul": xgb_test_pred,
        }
    )
    test_predictions["error"] = test_predictions["predicted_rul"] - test_predictions["true_rul"]
    test_predictions.to_csv(f"{MODELS_DIR}/test_predictions.csv", index=False)

    # ---------- 5. Report ----------
    print("\n" + "=" * 72)
    print(f"{'model':<15} {'val RMSE':>10} {'val NASA':>12} {'test RMSE':>10} {'test NASA':>12}")
    print("-" * 72)
    for name, r in results.items():
        print(
            f"{name:<15} {r['val']['rmse']:>10.2f} {r['val']['nasa_score']:>12.1f} "
            f"{r['test']['rmse']:>10.2f} {r['test']['nasa_score']:>12.1f}"
        )
    print("=" * 72)

    # ---------- 6. Sanity guard ----------
    # The brief: sane FD001 XGB/RF RMSE is usually low-teens to mid-20s.
    # If we're at 50-80, something upstream (labels, split, scaling) is broken.
    # If we're notably better than ~12, that's suspicious too -- it usually
    # means RUL_FD001.txt (the test labels) leaked into training somehow.
    xgb_test_rmse = results["xgboost"]["test"]["rmse"]
    if xgb_test_rmse > 35:
        print(
            f"\n*** WARNING: XGBoost test RMSE = {xgb_test_rmse:.1f}, higher than the "
            "expected low-teens-to-mid-20s range. Check the pipeline before shipping. ***"
        )
    elif xgb_test_rmse < 12:
        print(
            f"\n*** WARNING: XGBoost test RMSE = {xgb_test_rmse:.1f}, lower than the "
            "expected low-teens floor. Check for test-label leakage before shipping. ***"
        )
    else:
        print(f"\nXGBoost test RMSE = {xgb_test_rmse:.1f} -- within the expected sane range.")

    # ---------- 7. Save the headline model ----------
    joblib.dump(xgb, f"{MODELS_DIR}/xgb_model.joblib")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.joblib")
    with open(f"{MODELS_DIR}/feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(f"{MODELS_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{MODELS_DIR}/config_snapshot.json", "w") as f:
        json.dump(
            {
                "window_size": WINDOW_SIZE,
                "train_units": TRAIN_UNITS,
                "val_units": VAL_UNITS,
                "n_train_windows": int(len(X_train)),
                "n_val_windows": int(len(X_val)),
                "n_test_engines": int(len(X_test)),
                "mean_train_rul": mean_rul,
            },
            f,
            indent=2,
        )

    print(f"\nSaved model + scaler + metrics to {MODELS_DIR}/")
    print(f"Saved per-engine test predictions to {MODELS_DIR}/test_predictions.csv")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
