from statsmodels.tsa.arima.model import ARIMA

def train_arima_model(data, order=(1,1,1)):
    model = ARIMA(data, order=order)
    fitted_model = model.fit()
    return fitted_model

def train_arimax_model(data, exog, order=(1,1,1)):
    model = ARIMA(data, exog=exog, order=order)
    fitted_model = model.fit()
    return fitted_model
