import numpy as np

def parameter_estimator(prices):
    returns = np.log(prices/prices.shift(1)).dropna()
    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    return mu, sigma
