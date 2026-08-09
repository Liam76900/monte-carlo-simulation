import numpy as np
from option_pricing import price_european_call_mc

def compute_delta(S0, K, r, sigma, T, dt, Z, bump=0.01):
    price_up = price_european_call_mc(S0 * (1 + bump), K, r, sigma, T, dt, Z)
    price_down = price_european_call_mc(S0 * (1 - bump), K, r, sigma, T, dt, Z)
    
    delta = (price_up - price_down) / (2 * S0 * bump)
    return delta