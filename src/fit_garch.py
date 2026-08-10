from arch import arch_model

def fit_garch(historical_returns):
    returns_pct = historical_returns * 100
    model = arch_model(returns_pct, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
    fitted = model.fit(disp='off')
    return fitted