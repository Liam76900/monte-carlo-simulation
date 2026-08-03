# Monte Carlo Simulation Engine for Option Pricing & Risk Analysis

## Introduction

A python-based Monte Carlo framwork that uses AAPL's historical stock price data then prices a European call option, validates every component against known closed-form theory, and estimates risk (VaR/Expected Shortfall) two autonomous paths-from simulation and from historical returns-to cross check the model against reality.

## Overview

This project answers three principal and interlinked ideas about a stock (using 2020-2024 AAPL stock data):

1. How might this stock's price shift in the future?-simulated using Geometric Brownian Motion (GBM), adjusted on real historical.
2. What is a European call option on this stock worth today?-priced via Monte Carlo simulation, and cross-checked against the exact closed-form Black Scholes formula.
3. How much could an investor in this stock lose?-estimated via Value at Risk (VaR) and Expected Shortfall (ES), computed from the simulation and also from real historical returns and the comparison plotted.

All results are validated against an independent theoretical or historical benchmark, therefore, being a simulation with built-in proof

## Explanation of the Whole Project

This is logical chain of steps that I took to formulate this whole project:

1. The issue: Nobody can predict a stock's exact future price
A stock's future price is random and heavily influenced by many things, however, we can model how it tends to behave-using its average growth rate (drift) and how much it wobbles around that average (volatility). The standard model for this is the Geometric Borwnian Motion (GBM):
dS_t = μ S_t dt + σ S_t dW_t
In words: the change in price depends on an expected growth component (drift) plus a random shock component (volatility × randomness).

2. Estimating the model's inputs from real data- parameters.py
Instead of guessing the drift and volatility, we estimate them from AAPL's actual historical prices:
```python
def parameter_estimator(prices):
    returns = np.log(prices/prices.shift(1)).dropna()
    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    return mu, sigma
```
We take daily log returns, then scale by 252, which is the number of trading days in one year, to annualise them. mu is AAPL's estimated real-world expected growth rate; sigma is its estimated volatility

3. Generating randomness- generating_shock.py
GBM needs a source of randomness(dW_t in the formula above). This is generated as draws from a standard normal distribution (mean 0, std 1), one per simulated path per time step. My project also uses antithetic variates as a variation reduction technique-for every random draw Z, its mirror image of -Z is also used, which reduces noise in the final average without needing extra simulations.

4. Turning randomness into price paths- simulation_engine.py
```python
def simulate_gbm(S0, mu, sigma, T, dt, Z):
    drift = (mu - 0.5 * sigma**2) * dt
    shock = sigma * np.sqrt(dt) * Z
    log_returns = drift + shock
    price_paths = S0 * np.exp(np.cumsum(log_returns, axis=0))
    return price_paths
```
This is the one function the whole project builds off of, such as the pricing and risk calculations, and acts as the "engine" of the whole project.

Important distinction used throughout this project: when simulating realistic future price behaviour we use mu as we want to reflect what the stock could realistically do. When pricing an option, we use r, which is the risk-free rate, as the option pricing theory requires a risk neutral assumption for the resulting to be theoretically valid. Using the wrong one in either context was the biggest source of bugs wehn creating this project as I will discuss below.

5. Validating that the engine works- theoretical_mean (in validation.py)
Before any pricing or risk result, we have to check that the simulation engine itself is generating statistically correct behaviour, independant of options or payoffs.
GBM has a known formula for the expected (average) price at time T:
```python
def theoretical_mean(S0, mu, T):
    return S0 * np.exp(mu * T)
```
This states that "if the stock grows at rate mu, its average price after time T should be s0 * e^(mu*T)." So then we compare this theoretical number against the actual average of all our simulated final prices.
```python
print("Validation:")
print(f"Theoretical Mean: {theoretical}")
print(f"Simulated Mean: {simulated}")
```
If the numbers are similar this means that the simulation engine is working as it should.

6. Pricing the option- discounted_payoff.py adn option_pricing.py
An option's payoff only depends on the price at expiry (path[-1], the final simulated price on each path). For a call option:
```python
payoff = np.maximum(paths[-1] - K, 0)
```
You make profit if the price ends up above the strike K; otherwise the payoff is zero as you would not execute the option. Since this payoff occurs in the future, the payoff needs to be discounted back to today's value as money in the future is worth less than the present as it could have been earning risk-free interest in the meantime:
```python
discounted_payoff = np.exp(-r * T) * payoff
```
np.exp(-r * T) is the discount factor and helps to shrink the future value donw to the present value and averaging this discounted payoff across all simulated paths gives the Monte Carlo estimate of the option's fair price today.

7. Checking the option price is correct- black_scholes_call.py
A European call option also has an exact, closed-form solution-the Black-Scholes formula:
```python
def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return price
```
Comparing the Monte Carlo price against this exact formula-and watching these two values converge as the number of paths increases-is the strongest evidence that both the simulation engine and the option pricing logic are working properly together. See the Covergence Plot in Validation & Results below.

8. Quantifying how much to trust the estimate- confidence_interval.py
Every Monte Carlo run involves randomness, so the price estimate has some uncertainty with it. A confidence interval quantifies this: "we are 95% confident the true pric elies between X and Y."
```python
def confidence_interval(values, alpha=0.05):
    num = len(values)
    sd = np.std(values,ddof=1)
    mean = np.mean(values)
    z = norm.ppf(1-(alpha/2))
    confidence_interval_upper = mean + (z * sd/np.sqrt(num))
    confidence_interval_lower = mean - (z * sd/np.sqrt(num))
    return mean, (confidence_interval_lower, confidence_interval_upper)
```
This takes any array of simulated values and returns the mean plus its confidence bounds. As the number of paths of increases, this interval visibly narrows-more simulations mean a more reliable estimate

9. Meauring Risk- simulated_var_es.py and historical_var_es.py
Here this ansers another question: "if you hold this stock, how much could you lose?"

- Value at Risk(VaR): the loss threshold you should not expect to exceed most of the time (e.g. 95% of the time)
- Expected shortfall(ES): if a loss does exceed VaR, what is the average loss in those worst-case scenarios? ES captures tail risk that VaR alone can miss.
This is computed in two ways:
From the simulation:
```python
def compute_var_es(final_prices, S0, alpha=5):
    returns = (final_prices - S0) / S0
    value_at_risk = -np.percentile(returns, alpha)
    expected_shortfall = -returns[returns <= value_at_risk].mean()
    return value_at_risk, expected_shortfall
```

And from real historical AAPL returns:
```python
def compute_historical_var_es(historical_returns, alpha=5):
    value_at_risk = -np.percentile(historical_returns, alpha)
    expected_shortfall = -historical_returns[historical_returns <= value_at_risk].mean()
    return value_at_risk, expected_shortfall
```
Critical detail: both need to measure the same time horizon to be comparable. Historical returns here are daily; so the Monte Carlo simulation used for VaR is run over a single day (T = 1/252), using a separate simulation from the one used for option pricing (which correctly uses a full year, T = 1, matching the option's expiry). Comparing a 1-day risk estimate against a 1-year risk estimate was an early bug in this project — see Lessons Learned.

If the historical and Monte Carlo VaR/ES values land close together, it's evidence that the GBM model's assumptions (constant volatility, normally distributed daily returns) are a reasonable approximation of how AAPL actually behaves.

## Project Structure

monte-carlo-simulation/
├── src/
│   ├── parameters.py             # Estimates mu, sigma from historical prices
│   ├── generating_shock.py        # Generates random shocks (with antithetic variates)
│   ├── simulation_engine.py         # simulate_gbm — the core path simulation
│   ├── discounted_payoff.py           # Discounted option payoffs from simulated paths
│   ├── option_pricing.py                # price_european_call_mc — averages discounted payoffs
│   ├── black_scholes_call.py              # Closed-form theoretical option price
│   ├── confidence_interval.py               # CI for any array of Monte Carlo values
│   ├── validation.py                          # theoretical_mean — validates the simulation engine
│   ├── simulated_var_es.py                      # VaR/ES from simulated final prices
│   └── historical_var_es.py                       # VaR/ES from real historical returns
├── outputs/
│   ├── price_paths.png
│   ├── final_price_distribution.png
│   ├── convergence_plot.png
│   └── var_comparison.png
├── main.py                             # Runs the full pipeline end-to-end
├── requirements.txt
├── .gitignore
└── README.md

## Features

### Parameter Estimation (parameters.py)

- Estimates annualised drift (mu) and volatility (sigma) from real historical AAPL price data

### Simulation Engine (generating_shock.py, simulation_engine.py)

- Random shock generation with antithetic variates (variance reduction)
- GBM-based path simulation, reused by every downstream calculation

### Option Pricing (discounted_payoff.py, option_pricing.py, black_scholes_call.py)

- Monte Carlo pricing of a European call option
- Closed-form Black-Scholes price for direct comparison

### Validation (validation.py, confidence_interval.py)

- theoretical_mean: checks the simulation engine's statistical behaviour, independent of option pricing
- Confidence intervals on the Monte Carlo price estimate
- Convergence plot: MC price (with shaded 95% CI) vs. number of simulated paths, benchmarked against the theoretical Black-Scholes price

### Risk Metrics (simulated_var_es.py, historical_var_es.py)

- Monte Carlo VaR and Expected Shortfall (1-day horizon, from simulation)
- Historical VaR and Expected Shortfall (1-day horizon, from real AAPL returns)
- Bar chart comparing both, at matching time horizons

### Visualisations

- Simulated price paths over the option's 1-year horizon
- Histogram of final simulated prices (lognormal distribution)
- Convergence plot with shrinking confidence interval band
- Historical vs. Monte Carlo VaR/ES comparison bar chart

## Mathematical Background

### Asset price process (Geometric Brownian Motion):

dS_t = μ S_t dt + σ S_t dW_t

Discrete-time simulation formula (log-price):

### log(S_t) = log(S_0) + (μ - 0.5σ²)t + σ√t · Z,   Z ~ N(0,1)

### Option price (risk-neutral discounted expectation):

Price = e^(-rT) · E[max(S_T - K, 0)]

### Expected price under GBM (used for engine validation):

E[S_T] = S0 · e^(μT)

### Confidence interval on a Monte Carlo estimate:

CI = mean ± z · (std / √n)

## Example Output

Paramters:
S0: 190.3750762939453, mu: 0.2426480991324528, sigma: 0.3355336764080089
Monte Carlo VaR (95%): 0.0334, ES: 0.0019
Historical VaR (95%):  0.0324, ES: 0.0014
Validation:
Theoretical Mean: 242.65588075530093
Simulated Mean: 242.81530776772402
Option Pricing:
Monte Carlo Call Price: 28.161997863127183
Theoretical (Black-Scholes) Call Price: 27.9016
Difference: 0.2604
Discounted Payoffs: [ 82.55561408   0.           0.          49.51660918 141.34888109
   0.          32.63762283  20.86969024   0.          40.79647784]
Confidence Interval:
95% Confidence Interval:[27.8664, 28.4576]

## Installation
```bash
git clone <your-repo-url>
cd monte-carlo-simulation
pip install -r requirements.txt
```
Requirements: numpy, scipy, matplotlib, pandas, yfinance

## Usage
```bash
python main.py
```
This downloads AAPL historical data, estimates model parameters, runs the Monte Carlo simulation, prices the option, computes risk metrics, and displays all four plots described below.

## Validation & Results

1. ![Simulated Price Paths](outputs/price_paths.png)
Simulated price paths — 20 sample paths shown over the option's 1-year horizon, starting from AAPL's actual last price. Paths fan out over time as uncertainty compounds, as expected under GBM.

2. ![Final Price Distribution](outputs/final_price_distribution.png)
Final price distribution — a histogram of all simulated final prices, correctly right-skewed (lognormal), never going negative — the expected shape under GBM.

3. ![Convergence Plot](outputs/convergence_plot.png)
Convergence plot — Monte Carlo option price plotted against number of simulated paths (log scale), with a shaded 95% CI band, benchmarked against the theoretical Black-Scholes price. The MC price converges onto the theoretical price by ~5,000-10,000 paths, and the CI band visibly narrows as paths increase — direct evidence the pricing pipeline is correct.

4. ![Historical vs Monte Carlo VaR](outputs/var_comparison.png)
Historical vs. Monte Carlo VaR/ES — bar chart comparing both metrics at a matching 1-day horizon. Historical (~3.2% VaR) and Monte Carlo (~3.4% VaR) land close together, with Monte Carlo slightly higher — indicating the GBM model's risk estimate is a reasonable, if slightly more conservative, approximation of AAPL's real historical risk.

## Lessons Learned (bugs found and fixed during development)

- Drift confusion (mu vs r): early versions accidentally mixed up the real-world drift (mu) and the risk-free rate (r). Simulating realistic price behavior requires mu; pricing an option requires r for the result to be theoretically valid. Using the wrong one in either place causes systematic mispricing.
- Hardcoded comparison parameters: the theoretical price was initially computed with hardcoded placeholder values (S0=100, K=100, sigma=0.2) instead of the actual estimated parameters, making the convergence plot's benchmark line meaningless. Fixed by passing the same real S0, K, r, sigma into both the Monte Carlo and Black-Scholes calculations.
- Time horizon mismatch in VaR comparison: historical returns are daily, but the first Monte Carlo VaR used a full year of simulated growth — comparing a 1-day risk figure against a 1-year one produced wildly different (and meaningless) results. Fixed by running a separate, short (1-day) simulation specifically for the VaR comparison, kept independent from the 1-year simulation used for option pricing.