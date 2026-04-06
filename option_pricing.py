import numpy as np

def price_european_call_mc(S0, K, r, sigma, T, dt, Z):
    drift = (r - 0.5 * sigma**2) * dt
    log_returns = drift + sigma * np.sqrt(dt) * Z
    paths = S0 * np.exp(np.cumsum(log_returns, axis=0))
    payoff = np.maximum(paths[-1] - K, 0)
    price = np.exp(-r * T) * np.mean(payoff)
    return price