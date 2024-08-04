def predict_arima(model, steps=2):
    forecast = model.get_forecast(steps)
    ypred = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)
    return ypred, conf_int

def predict_arimax(model, exog, steps=2):
    forecast = model.get_forecast(steps, exog=exog)
    ypred = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)
    return ypred, conf_int
