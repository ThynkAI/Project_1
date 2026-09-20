"""
Central place for every constant the pipeline uses.
Change a number here, not in five different scripts.
"""

# --- Raw column layout (fixed by the C-MAPSS file format) ---
INDEX_NAMES = ["unit_number", "time_cycles"]
SETTING_NAMES = ["setting_1", "setting_2", "setting_3"]
SENSOR_NAMES = [f"sensor_{i}" for i in range(1, 22)]  # sensor_1 .. sensor_21
ALL_COLUMNS = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES  # 26 columns total

# --- Sensors/settings that are (near-)constant on FD001 ---
# Found by checking std/nunique on the training data (see notebooks/eda.py).
# These 7 columns have exactly one unique value across all 20,631 rows,
# so they carry zero predictive signal and only add noise/collinearity.
DROP_COLUMNS = [
    "setting_3",
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]

# What's left after dropping the dead columns: 2 settings + 15 sensors = 17 features
FEATURE_COLUMNS = [c for c in SETTING_NAMES + SENSOR_NAMES if c not in DROP_COLUMNS]

# --- Labeling ---
# Piecewise-linear RUL target: an engine is treated as "fully healthy" (flat RUL)
# until it's within RUL_CLIP cycles of failure, after which RUL decays 1:1 with
# cycle count. This is the standard C-MAPSS convention (Heimes 2008, and widely
# used since) because early-life degradation isn't linear/visible in the sensors,
# so asking a model to distinguish RUL=290 from RUL=250 just teaches it noise.
RUL_CLIP = 125

# --- Windowing ---
# Each training example = one 30-cycle sliding window of sensor history,
# labeled with the RUL at the window's *last* cycle.
WINDOW_SIZE = 30

# Per-window summary statistics computed for every retained feature column.
# Chosen to be cheap, interpretable, and to capture both the current sensor
# level (mean/last) and the degradation trend (slope) within the window.
WINDOW_STATS = ["mean", "std", "min", "max", "last", "slope"]

# --- Train/validation split ---
# Split by ENGINE ID, never by row. A random row split would leak cycles from
# the same engine's failure trajectory into both train and validation, which
# lets the model "memorize" an engine instead of generalizing to a new one.
TRAIN_UNITS = list(range(1, 81))    # engines 1-80 -> training
VAL_UNITS = list(range(81, 101))    # engines 81-100 -> validation

# --- Reproducibility ---
RANDOM_STATE = 42
