import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_time_series(data):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x='Date', y='AAPL')
    plt.title('Apple Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.xticks(rotation=90)
    plt.show()

def plot_decomposition(decomposed):
    trend = decomposed.trend
    seasonal = decomposed.seasonal
    residual = decomposed.resid

    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(decomposed.observed, label='Original', color='black')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(trend, label='Trend', color='red')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonal', color='blue')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(residual, label='Residual', color='black')
    plt.legend(loc='upper left')
    plt.show()

def plot_acf_pacf(series):
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    plot_acf(series.dropna(), ax=plt.gca())
    plt.title('ACF Plot')
    plt.subplot(122)
    plot_pacf(series.dropna(), lags=11, ax=plt.gca())
    plt.title('PACF Plot')
    plt.show()

def plot_differenced_series(differenced_series):
    plt.figure(figsize=(12, 6))
    plt.plot(differenced_series)
    plt.title('Differenced Series')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.xticks(rotation=30)
    plt.show()

def plot_arima_forecast(data, forecast, ypred, conf_int):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['AAPL'], label='Actual')
    plt.plot(forecast.predicted_mean.index, ypred, color='orange', label='Forecast')
    plt.fill_between(forecast.predicted_mean.index,
                     conf_int['lower AAPL'],
                     conf_int['upper AAPL'],
                     color='k', alpha=.15)
    plt.title('ARIMA Forecast')
    plt.legend(loc='lower right')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.xticks(rotation=30)
    plt.show()

def plot_arimax_forecast(data, forecast, ypred, conf_int):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['AAPL'], label='Actual')
    plt.plot(forecast.predicted_mean.index, ypred, color='orange', label='Forecast')
    plt.fill_between(forecast.predicted_mean.index,
                     conf_int['lower AAPL'],
                     conf_int['upper AAPL'],
                     color='k', alpha=.15)
    plt.title('ARIMAX Forecast')
    plt.legend(loc='lower right')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.xticks(rotation=30)
    plt.show()

def plot_training_testing_data(train, test):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Close'], label='Training Data')
    plt.plot(test.index, test['Close'], label='Testing Data', color='orange')
    plt.title('Training and Testing Data')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend(loc='upper left')
    plt.xticks(rotation=30)
    plt.show()

def plot_feature_importances(model):
    importances = model.feature_importances_
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances, color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title('Feature Importances from XGBoost Model')
    plt.show()

def plot_backtest_results(predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(predictions.index, predictions['Target'], label='Actual')
    plt.plot(predictions.index, predictions['predictions'], label='Predicted', color='orange')
    plt.title('Backtest Results')
    plt.xlabel('Date')
    plt.ylabel('Target')
    plt.legend(loc='upper left')
    plt.show()
