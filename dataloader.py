import yfinance as yf
import numpy as np

def dataloader(ticker)
    data = yf.download(ticker, start="2020-01-01", end="2024-01-01")
    prices = data["Adj Close"]
    returns = np.log(prices/prices.shift(1)).dropna()