import numpy as np

def compute_var_es(final_prices, S0, alpha=5):
    returns = (final_prices - S0) / S0
    value_at_risk = np.percentile(final_prices, alpha)
    expected_shortfall = returns[returns <= value_at_risk].mean()
    return value_at_risk, expected_shortfall