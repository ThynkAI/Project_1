"""
Load the raw C-MAPSS FD001 text files and turn them into clean,
labeled DataFrames. No modeling happens here -- just I/O and labeling.
"""

import pandas as pd

from config import ALL_COLUMNS, DROP_COLUMNS, RUL_CLIP


def load_raw(path: str) -> pd.DataFrame:
    """
    Load a raw C-MAPSS space-separated file (train_FD001.txt or test_FD001.txt)
    and attach the 26 standard column names.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLUMNS)
    return df


def load_rul(path: str) -> pd.Series:
    """
    Load RUL_FD001.txt: one true RUL value per test engine, in engine-ID order
    (row 0 = engine 1's true remaining life at the point its trajectory was cut).
    """
    rul = pd.read_csv(path, sep=r"\s+", header=None, names=["RUL"])
    rul.index = rul.index + 1  # 1-indexed to match unit_number
    return rul["RUL"]


def add_train_rul_labels(train_df: pd.DataFrame, clip: int = RUL_CLIP) -> pd.DataFrame:
    """
    Add a per-row 'RUL' column to the training data.

    Training engines run all the way to failure, so at any cycle we know
    exactly how many cycles are left: (that engine's final cycle) - (this cycle).
    We then clip at `clip` cycles because early-life degradation is not
    reliably visible in the sensors -- treating RUL as flat during the
    healthy plateau keeps the model from chasing noise.
    """
    df = train_df.copy()
    max_cycle_per_unit = df.groupby("unit_number")["time_cycles"].transform("max")
    raw_rul = max_cycle_per_unit - df["time_cycles"]
    df["RUL"] = raw_rul.clip(upper=clip)
    return df


def drop_dead_columns(df: pd.DataFrame, drop_columns=DROP_COLUMNS) -> pd.DataFrame:
    """
    Drop settings/sensors that are constant (zero std, one unique value) on
    the training data. These add no signal and only waste model capacity /
    add numerical noise once scaled.
    """
    return df.drop(columns=drop_columns)


def prepare_train(path: str) -> pd.DataFrame:
    """Full prep for the training file: load -> label -> drop dead columns."""
    df = load_raw(path)
    df = add_train_rul_labels(df)
    df = drop_dead_columns(df)
    return df


def prepare_test(path: str) -> pd.DataFrame:
    """Full prep for the test file: load -> drop dead columns (no labels; truncated trajectories)."""
    df = load_raw(path)
    df = drop_dead_columns(df)
    return df


if __name__ == "__main__":
    # Quick sanity check when run directly: python data_prep.py
    train = prepare_train("../data/train_FD001.txt")
    test = prepare_test("../data/test_FD001.txt")
    rul = load_rul("../data/RUL_FD001.txt")

    print("train shape:", train.shape)
    print("test shape:", test.shape)
    print("rul shape:", rul.shape)
    print("train RUL range:", train["RUL"].min(), "-", train["RUL"].max())
    print("train columns:", list(train.columns))
