import numpy as np

def get_annualized_volatility(fitted):
    return fitted.conditional_volatility * np.sqrt(252) / 100