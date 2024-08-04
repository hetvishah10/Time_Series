import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, precision_score
from xgboost import XGBClassifier

# Load the dataset from yfinance
data = yf.download("AAPL", start="2000-01-01", end="2022-05-31")
data['Next_day'] = data['Close'].shift(-1)
data['Target'] = (data['Next_day'] > data['Close']).astype(int)

# Handle missing values
data.dropna(inplace=True)

# Univariate Analysis - Apple stock prices
df = data[['Close']].rename(columns={'Close': 'AAPL'})
df = df.set_index(data.index)

# Ensure index is a DatetimeIndex with frequency set
df.index = pd.DatetimeIndex(df.index).to_period('D').to_timestamp()

# Plotting the Apple stock prices
sns.lineplot(data=df, x=df.index, y='AAPL')
plt.xticks(rotation=90)
plt.title('Apple Stock Prices')
plt.show()

# Decomposition
decomposed = seasonal_decompose(df['AAPL'], period=365)
trend = decomposed.trend
seasonal = decomposed.seasonal
residual = decomposed.resid

plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(df['AAPL'], label='Original', color='black')
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

# ACF and PACF plots
plt.rcParams.update({'figure.figsize':(7,4), 'figure.dpi':80})
plot_acf(df['AAPL'])
plt.show()
plot_pacf(df['AAPL'], lags=11)
plt.show()

# Augmented Dickey-Fuller (ADF) Test
results = adfuller(df['AAPL'])
print('ADF p-value:', results[1])

# First order differencing
v1 = df['AAPL'].diff().dropna()
results1 = adfuller(v1)
print('ADF p-value after differencing:', results1[1])

# Plot differenced series
plt.plot(v1)
plt.title('1st Order Differenced Series')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.xticks(rotation=30)
plt.show()

# ARIMA Model
arima = ARIMA(df['AAPL'], order=(1,1,1))
ar_model = arima.fit()
print(ar_model.summary())

# Forecast
forecast = ar_model.get_forecast(2)
ypred = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.05)

# Create DataFrame for predictions
dp = pd.DataFrame({
    'Date': pd.to_datetime(['2024-01-01', '2024-02-01']),
    'price_actual': ['184.40', '185.04'],
    'price_predicted': ypred.values,
    'lower_int': conf_int['lower AAPL'].values,
    'upper_int': conf_int['upper AAPL'].values
})
dp.set_index('Date', inplace=True)

# Plotting ARIMA predictions
plt.plot(df['AAPL'], label='Actual')
plt.plot(dp['price_predicted'], color='orange', label='Predicted')
plt.fill_between(dp.index, dp['lower_int'], dp['upper_int'], color='k', alpha=.15)
plt.title('ARIMA Model Performance')
plt.legend(loc='lower right')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.xticks(rotation=30)
plt.show()

# Evaluate ARIMA Model
print('ARIMA MAE:', mean_absolute_error(dp['price_actual'].astype(float), dp['price_predicted']))

# ARIMAX Model with exogenous variable
dfx = data[['Close', 'Volume']].rename(columns={'Close': 'AAPL', 'Volume': 'TXN'})
dfx.set_index(data.index, inplace=True)

# Ensure index is a DatetimeIndex with frequency set
dfx.index = pd.DatetimeIndex(dfx.index).to_period('D').to_timestamp()

model2 = ARIMA(dfx['AAPL'], exog=dfx['TXN'], order=(1,1,1))
arimax = model2.fit()
print(arimax.summary())

# Forecast with ARIMAX
forecast = arimax.get_forecast(2, exog=dfx['TXN'].iloc[-2:].values)
ypred = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.05)

# Create DataFrame for ARIMAX predictions
dpx = pd.DataFrame({
    'Date': pd.to_datetime(['2024-01-01', '2024-02-01']),
    'price_actual': ['184.40', '185.04'],
    'price_predicted': ypred.values,
    'lower_int': conf_int['lower AAPL'].values,
    'upper_int': conf_int['upper AAPL'].values
})
dpx.set_index('Date', inplace=True)

# Plotting ARIMAX predictions
plt.plot(df['AAPL'], label='Actual')
plt.plot(dpx['price_predicted'], color='orange', label='Predicted')
plt.fill_between(dpx.index, dpx['lower_int'], dpx['upper_int'], color='k', alpha=.15)
plt.title('ARIMAX Model Performance')
plt.legend(loc='lower right')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.xticks(rotation=30)
plt.show()

# Evaluate ARIMAX Model
print('ARIMAX MAE:', mean_absolute_error(dpx['price_actual'].astype(float), dpx['price_predicted']))

# XGBoost Model
train = data.iloc[:-30]
test = data.iloc[-30:]
features = ['Open', 'High', 'Low', 'Close', 'Volume']

model1 = XGBClassifier(max_depth=3, n_estimators=100, random_state=42)
model1.fit(train[features], train['Target'])
model1_preds = model1.predict(test[features])

# Evaluate XGBoost Model
model1_preds = pd.Series(model1_preds, index=test.index)
plt.plot(test['Target'], label='Actual')
plt.plot(model1_preds, label='Predicted')
plt.legend()
plt.show()

# Backtesting function
def predict(train, test, features, model):
    model.fit(train[features], train['Target'])
    model_preds = model.predict(test[features])
    model_preds = pd.Series(model_preds, index=test.index, name='predictions')
    combine = pd.concat([test['Target'], model_preds], axis=1)
    return combine

def backtest(data, model, features, start=5031, step=120):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[:i].copy()
        test = data.iloc[i:(i+step)].copy()
        model_preds = predict(train, test, features, model)
        all_predictions.append(model_preds)
    return pd.concat(all_predictions)

# Backtesting
predictions = backtest(data, model1, features)
print(predictions)
print('Backtest Precision Score:', precision_score(predictions['Target'], predictions['predictions']))

