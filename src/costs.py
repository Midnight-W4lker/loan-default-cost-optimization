"""Validation-only decision-threshold optimization."""

import numpy as np
import pandas as pd


def threshold_costs(y_true, probability, false_negative_cost=10.0, false_positive_cost=1.0):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Inputs must be aligned binary labels and probabilities.")
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        predicted = p >= threshold
        fn = int(((y == 1) & ~predicted).sum())
        fp = int(((y == 0) & predicted).sum())
        rows.append({"threshold": threshold, "false_negatives": fn, "false_positives": fp, "total_cost": false_negative_cost * fn + false_positive_cost * fp})
    return pd.DataFrame(rows)


def optimal_threshold(y_true, probability, false_negative_cost=10.0, false_positive_cost=1.0):
    table = threshold_costs(y_true, probability, false_negative_cost, false_positive_cost)
    winner = table.loc[table["total_cost"].idxmin()]
    return float(winner["threshold"]), table

