"""
Turn per-cycle sensor rows into per-window feature vectors.

A "window" is WINDOW_SIZE consecutive cycles from one engine. Instead of
feeding raw timesteps to a tree model (which can't natively use sequence
order), we summarize each window into a handful of statistics per sensor:
its level (mean/last), its spread (std/min/max), and its trend (slope).
That turns a sequence problem into a tabular one that Ridge/RF/XGBoost can use.
"""

import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, WINDOW_SIZE


def _slope(values: np.ndarray) -> float:
    """Least-squares slope of `values` against cycle index 0..n-1 within the window."""
    x = np.arange(len(values))
    # np.polyfit is fine at this scale (windows are only 30 points)
    return float(np.polyfit(x, values, 1)[0])


def _window_stats(window: pd.DataFrame, columns=FEATURE_COLUMNS) -> dict:
    """Compute mean/std/min/max/last/slope for every feature column in one window."""
    stats = {}
    for col in columns:
        vals = window[col].to_numpy()
        stats[f"{col}_mean"] = vals.mean()
        stats[f"{col}_std"] = vals.std()
        stats[f"{col}_min"] = vals.min()
        stats[f"{col}_max"] = vals.max()
        stats[f"{col}_last"] = vals[-1]
        stats[f"{col}_slope"] = _slope(vals)
    return stats


def build_training_windows(
    df: pd.DataFrame,
    unit_ids,
    window_size: int = WINDOW_SIZE,
    columns=FEATURE_COLUMNS,
) -> pd.DataFrame:
    """
    Slide a window of size `window_size` (stride 1) over every engine in
    `unit_ids` and emit one row of stats + label per window.

    Only engines in `unit_ids` are used, so this same function builds both
    the train split (units 1-80) and the validation split (units 81-100)
    just by changing which IDs are passed in.
    """
    rows = []
    for unit in unit_ids:
        engine_df = df[df["unit_number"] == unit].sort_values("time_cycles")
        n = len(engine_df)
        if n < window_size:
            # Shouldn't happen on FD001 (shortest train engine has 128 cycles)
            # but guard anyway rather than silently skipping.
            raise ValueError(f"Engine {unit} has only {n} cycles, need >= {window_size}")

        for start in range(0, n - window_size + 1):
            window = engine_df.iloc[start : start + window_size]
            stats = _window_stats(window, columns)
            stats["unit_number"] = unit
            stats["last_cycle"] = int(window["time_cycles"].iloc[-1])
            stats["RUL"] = float(window["RUL"].iloc[-1])
            rows.append(stats)

    return pd.DataFrame(rows)


def build_last_window_per_engine(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    columns=FEATURE_COLUMNS,
) -> pd.DataFrame:
    """
    For each engine, take ONLY its final `window_size` cycles and compute
    one stats row. This is how the official test score is built: each test
    engine's trajectory is already truncated at some random point, and the
    question is "how many cycles are left right now" -- i.e. at the last
    row we have, not at every historical point.
    """
    rows = []
    for unit, engine_df in df.groupby("unit_number"):
        engine_df = engine_df.sort_values("time_cycles")
        window = engine_df.tail(window_size)
        stats = _window_stats(window, columns)
        stats["unit_number"] = unit
        stats["last_cycle"] = int(window["time_cycles"].iloc[-1])
        rows.append(stats)

    return pd.DataFrame(rows).sort_values("unit_number").reset_index(drop=True)


def stat_feature_names(columns=FEATURE_COLUMNS):
    """The ordered list of engineered feature-column names (used to align X at inference time)."""
    names = []
    for col in columns:
        for stat in ["mean", "std", "min", "max", "last", "slope"]:
            names.append(f"{col}_{stat}")
    return names


if __name__ == "__main__":
    from data_prep import prepare_train

    train = prepare_train("../data/train_FD001.txt")
    windows = build_training_windows(train, unit_ids=[1, 2])
    print("windows shape for engines 1-2:", windows.shape)
    print(windows[["unit_number", "last_cycle", "RUL"]].head())
    print(windows[["unit_number", "last_cycle", "RUL"]].tail())
