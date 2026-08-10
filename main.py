import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append('src')

from src.parameters import parameter_estimator
from src.generating_shock import generating_shock
from src.simulation_engine import simulate_gbm
from src.simulated_var_es import compute_var_es
from src.historical_var_es import compute_historical_var_es
from src.option_pricing import price_european_call_mc
from src.black_scholes_call import black_scholes_call
from src.validation import theoretical_mean
from src.discounted_payoff import discounted_payoffs
from src.confidence_interval import confidence_interval
from src.compute_delta import compute_delta
from src.black_scholes_delta import black_scholes_delta
from src.fit_garch import fit_garch
from src.compute_annualised_volatility import compute_annualised_volatility
from src.forecast_volatility import forecast_volatility

ticker="AAPL"
data = yf.download(ticker, start="2020-01-01", end="2024-01-01", auto_adjust=False)
data.columns = data.columns.droplevel(1)
prices = data["Adj Close"]
historical_returns = data['Close'].pct_change().dropna().values

mu, sigma = parameter_estimator(prices)
S0 = prices.iloc[-1]

T = 1
n_steps=252
dt = 1/252
N = 252
n_sim = 100000
r = 0.03
K = S0
theoretical_price = black_scholes_call(S0, K, r, sigma, T)
path_counts = [100, 500, 1000, 5000, 10000, 50000, 100000]
means, ci_lowers, ci_uppers = [], [], []

Z = generating_shock(N, n_sim, antithetic=True)

price_paths = simulate_gbm(S0, mu, sigma, T, dt, Z)
final_prices = price_paths[-1]

value_at_risk_5, expected_shortfall = compute_var_es(final_prices, S0)

theoretical = theoretical_mean(S0, mu, T)
simulated = np.mean(final_prices)

option_price = price_european_call_mc(S0, K , r, sigma, T, dt, Z)

discounted_payoff = discounted_payoffs(S0, K, r, sigma, T, dt, Z)

mean, (ci_lower, ci_upper) = confidence_interval(discounted_payoff)

mc_delta = compute_delta(S0, K, r, sigma, T, dt, Z)
bs_delta = black_scholes_delta(S0, K, r, sigma, T)

T_var = 1/252
Z_var = generating_shock(1, n_sim, antithetic=True)
price_paths_var = simulate_gbm(S0, mu, sigma, T_var, dt, Z_var)
final_prices_var = price_paths_var[-1]

print("Paramters:")
print(f"S0: {S0}, mu: {mu}, sigma: {sigma}")

# Monte Carlo
mc_var, mc_es = compute_var_es(final_prices_var, S0, alpha=5)

# Historical
hist_var, hist_es = compute_historical_var_es(historical_returns, alpha=5)

print(f"Monte Carlo VaR (95%): {mc_var:.4f}, ES: {mc_es:.4f}")
print(f"Historical VaR (95%):  {hist_var:.4f}, ES: {hist_es:.4f}")

# GARCH volatility analysis
fitted_garch = fit_garch(historical_returns)
garch_vol = compute_annualised_volatility(fitted_garch)
garch_forecast = forecast_volatility(fitted_garch, horizon=10)

print("Validation:")
print(f"Theoretical Mean: {theoretical}")
print(f"Simulated Mean: {simulated}")

print("Option Pricing:")
print(f"Monte Carlo Call Price: {option_price}")
print(f"Theoretical (Black-Scholes) Call Price: {theoretical_price:.4f}")
print(f"Difference: {abs(option_price - theoretical_price):.4f}")

print(f"Discounted Payoffs: {discounted_payoff[:10]}")

print("Confidence Interval:")
print(f"95% Confidence Interval:[{ci_lower:.4f}, {ci_upper:.4f}]")

print(f"Monte Carlo Delta: {mc_delta:.4f}")
print(f"Theoretical Delta: {bs_delta:.4f}")

print("GARCH Volatility Analysis:")
print(f"GBM constant sigma: {sigma:.4f}")
print(f"GARCH volatility range: {garch_vol.min():.4f} to {garch_vol.max():.4f}")
print(f"GARCH 10-day forward forecast: {garch_forecast}")

plt.plot(price_paths[:, :20])
plt.title("Monte Carlo Simulated Paths")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.savefig("price_paths.png", dpi=150, bbox_inches='tight')
plt.show()

plt.hist(final_prices, bins=50)
plt.title("Final Price Distribution")
plt.savefig("final_price_distribution.png", dpi=150, bbox_inches='tight')
plt.show()

labels = ['VaR (95%)', 'Expected Shortfall (95%)']
historical_values = [hist_var, hist_es]
mc_values = [mc_var, mc_es]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(7,5))
ax.bar(x - width/2, historical_values, width, label='Historical', color='steelblue')
ax.bar(x + width/2, mc_values, width, label='Monte Carlo', color='darkorange')

ax.set_ylabel('Loss (as % of value)')
ax.set_title('Historical vs Monte Carlo: VaR and Expected Shortfall')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.savefig("var_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

for n in path_counts:
    Z = np.random.standard_normal((n_steps, n))
    dp = discounted_payoffs(S0, K, r, sigma, T, dt, Z)
    mean, (lo, hi) = confidence_interval(dp, alpha=0.05)
    means.append(mean)
    ci_lowers.append(lo)
    ci_uppers.append(hi)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(path_counts, means, marker='o', label='MC price')
plt.fill_between(path_counts, ci_lowers, ci_uppers, alpha=0.2, label='95% CI')
plt.axhline(theoretical_price, color='red', linestyle='--', label='Theoretical price')
plt.xscale('log')
plt.xlabel('Number of paths (log scale)')
plt.ylabel('Option price')
plt.title('MC Price with Confidence Interval vs Path Count')
plt.legend()
plt.savefig("convergence_plot.png", dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(garch_vol)
plt.axhline(sigma, color='red', linestyle='--', label=f'Constant GBM sigma ({sigma:.4f})')
plt.title('GARCH-Estimated Volatility Over Time vs. Constant GBM Assumption')
plt.xlabel('Time')
plt.ylabel('Annualised Volatility')
plt.legend()
plt.savefig('outputs/garch_volatility.png', dpi=150, bbox_inches='tight')
plt.show()