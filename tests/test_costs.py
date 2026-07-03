import numpy as np

from src.costs import optimal_threshold, threshold_costs


def test_cost_formula():
    table = threshold_costs([0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9], 10, 1)
    row = table[np.isclose(table.threshold, 0.5)].iloc[0]
    assert row.false_negatives == 1 and row.false_positives == 1
    assert row.total_cost == 11


def test_higher_fn_cost_discourages_high_threshold():
    y = [0, 0, 0, 1, 1]
    p = [0.1, 0.2, 0.4, 0.3, 0.7]
    low_fn_threshold, _ = optimal_threshold(y, p, 1, 1)
    high_fn_threshold, _ = optimal_threshold(y, p, 20, 1)
    assert high_fn_threshold <= low_fn_threshold

