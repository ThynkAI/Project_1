"""
The two numbers every C-MAPSS result gets reported in:

- RMSE: plain root-mean-squared-error in cycles. Symmetric -- treats
  over- and under-predicting a failure the same.

- NASA PHM-2008 score: the scoring function from the original 2008
  PHM Society challenge (Saxena et al.). It's asymmetric on purpose:
  predicting MORE remaining life than an engine actually has (a "late"
  prediction) is penalized much harder than predicting less, because in
  real operations a late prediction means you skip maintenance the
  engine actually needed. Lower is still better; it's not a percentage
  or bounded, so it's only meaningful as a relative comparison between
  models on the same data.
"""

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """
    d = predicted - true
      d < 0  -> early prediction (predicted less remaining life than truth):
                score = exp(-d/13) - 1
      d >= 0 -> late prediction (predicted more remaining life than truth):
                score = exp(d/10)  - 1   <- grows faster (smaller denominator)
    Summed over all samples.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "n": len(y_true),
    }


if __name__ == "__main__":
    # Sanity checks against hand-computable cases
    y_true = np.array([100, 50, 10])
    y_pred = np.array([100, 50, 10])
    assert rmse(y_true, y_pred) == 0.0
    assert nasa_score(y_true, y_pred) == 0.0

    # Late prediction (over-estimate) should score worse than an equal-sized
    # early prediction (under-estimate)
    late = nasa_score(np.array([100]), np.array([110]))   # d = +10
    early = nasa_score(np.array([100]), np.array([90]))   # d = -10
    print("late (d=+10):", late, " early (d=-10):", early)
    assert late > early, "late predictions must be penalized harder than early ones"

    print("metrics.py sanity checks passed")
