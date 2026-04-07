import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

from parameters import parameter_estimator
from generating_shock import generating_shock
from simulation_engine import simulate_gbm
from risk_metrics import compute_var_es
from option_pricing import price_european_call_mc
from validation import theoretical_mean

ticker="AAPL"
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")
prices = data["Adj Close"]



