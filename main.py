import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from parameters import parameter_estimator
from generating_shock import generating_shock
from simulation_engine import simulate_gbm
from risk_metrics import compute_var_es
from option_pricing import price_european_call_mc
from validation import theoretical_mean

ticker="AAPL"
data = yf.download(ticker, start="2020-01-01", end="2024-01-01", auto_adjust=False)
data.columns = data.columns.droplevel(1)
prices = data["Adj Close"]

mu, sigma = parameter_estimator(prices)
S0 = prices.iloc[-1]

T = 1
dt = 1/252
N = int(T / dt)
n_sim = 10000
r = 0.03
K = S0

Z = generating_shock(N, n_sim, antithetic=True)

price_paths = simulate_gbm(S0, mu, sigma, T, dt, Z)
final_prices = price_paths[-1]

value_at_risk_5, expected_shortfall = compute_var_es(final_prices, S0)

theoretical = theoretical_mean(S0, mu, T)
simulated = np.mean(final_prices)

option_price = price_european_call_mc(S0, K , r, sigma, T, dt, Z)

print("Paramters:")
print(f"S0: {S0}, mu: {mu}, sigma: {sigma}")

print("/nRisk Metrics:")
print(f"Value at Risk (5%): {value_at_risk_5}")
print(f"Expected Shortfall: {expected_shortfall}")

print("Validation:")
print(f"Theoretical Mean: {theoretical}")
print(f"Simulated Mean: {simulated}")

print("/nOption Pricing:")
print(f"Monte Carlo Call Price: {option_price}")

plt.plot(price_paths[:, :20])
plt.title("Monte Carlo Simulated Paths")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.show()

plt.hist(final_prices, bins=50)
plt.title("Final Price Distribution")
plt.show()