import numpy as np

def compute_annualised_volatility(fitted):
    return fitted.conditional_volatility * np.sqrt(252) / 100