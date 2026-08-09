import numpy as np

def compute_var_es(final_prices, S0, alpha=5):
    returns = (final_prices - S0) / S0
    raw_percentile = np.percentile(returns, alpha)   # negative number, e.g. -0.0334
    value_at_risk = -raw_percentile                   # flipped to positive, e.g. 0.0334

    tail = returns[returns <= raw_percentile]          # filter using the ORIGINAL negative threshold
    expected_shortfall = -tail.mean()                   # flip the tail average to positive too

    return value_at_risk, expected_shortfall