import numpy as np

def compute_historical_var_es(historical_returns, alpha=5):
    raw_percentile = np.percentile(historical_returns, alpha)
    value_at_risk = -raw_percentile

    tail = historical_returns[historical_returns <= raw_percentile]
    expected_shortfall = -tail.mean()

    return value_at_risk, expected_shortfall