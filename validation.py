import numpy as np

def theoretical(S0, mu, T):
    return S0 * np.exp(mu * T)