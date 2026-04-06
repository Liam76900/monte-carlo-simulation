import numpy as np

def compute_var_es(final_prices, alpha=5):
    value_at_risk = np.percentile(final_prices, alpha)
    expected_shortfall = final_prices[final_prices <= value_at_risk].mean()
    return value_at_risk, expected_shortfall