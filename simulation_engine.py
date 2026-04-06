import numpy as np

def simulate_gbm(S0, mu, sigma, T, dt, Z):
    drift = (mu - 0.5 * sigma**2) * dt
    shock = sigma * np.sqrt(dt) * Z
    log_returns = drift + shock
    price_paths = S0 * np.exp(np.cumsum(log_returns, axis=0))
    return price_paths