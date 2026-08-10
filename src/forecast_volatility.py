import numpy as np

def forecast_volatility(fitted, horizon=10):
    forecast = fitted.forecast(horizon=horizon)
    forecast_variance = forecast.variance.values[-1]
    return np.sqrt(forecast_variance) * np.sqrt(252) / 100