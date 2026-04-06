import numpy as np

def generating_shock(N, n_sim, antithetic=True):
    Z = np.random.standard.normal((N, n_sim // 2 if antithetic else n_sim))
    if antithetic:
        Z = np.concatenate([Z, -Z], axis=1)
    return Z