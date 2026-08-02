import numpy as np

def compute_historical_var_es(historical_returns, alpha=5):
    value_at_risk = -np.percentile(historical_returns, alpha)
    expected_shortfall = -historical_returns[historical_returns <= value_at_risk].mean()
    return value_at_risk, expected_shortfall