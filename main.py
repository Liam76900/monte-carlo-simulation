import yfinance as yf
import numpy as np

ticker="AAPL"
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")
prices = data["Adj Close"]