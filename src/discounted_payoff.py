import numpy as np
from simulation_engine import simulate_gbm

def discounted_payoffs(S0, K, r, sigma, T, dt, Z):
    paths = simulate_gbm(S0, r, sigma, T, dt, Z)
    payoff = np.maximum(paths[-1] - K, 0)
    discounted_payoff = np.exp(-r * T) * payoff
    return discounted_payoff