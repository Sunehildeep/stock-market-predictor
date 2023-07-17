import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import yfinance as yf
from datetime import datetime, timedelta

# Define past_years
past_years = 5

ticker = 'TTWO'
msft = yf.Ticker(ticker)

t_now = datetime.now()
t_prev = t_now - timedelta(days=past_years * 365)

data = yf.download(ticker, start=t_prev, end=t_now, progress=False)


data['Date'] = data.index.values

# Feature Engineering
# You can include additional features like technical indicators here if needed
data = data[['Close', 'Adj Close']]

# Data Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# Prepare Data for LSTM
lookback = 60  # Number of previous days to consider as input for prediction
X = []
y = []

for i in range(len(scaled_data) - lookback):
    X.append(scaled_data[i:i+lookback])
    y.append(scaled_data[i+lookback, 1])  # Using the 'Adj Close' as the target variable
X, y = np.array(X), np.array(y)

# Train-Test Split
split = int(0.8 * len(X))  # 80% train, 20% test
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

#  Reshape Data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

#  Build LSTM Model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(lookback, 2)))
model.add(LSTM(units=100, return_sequences=True))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# Train the LSTM Model
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Predict Future Stock Prices
X_current = scaled_data[-lookback:].reshape(1, -1, 2)  # Last lookback days in the available data
current_prices = []

num_days = 30  # Number of days to predict into the future
for _ in range(num_days):
    current_price = model.predict(X_current)
    current_prices.append(current_price[0, 0])
    X_current = np.roll(X_current, -1)  # Shift the input sequence by one day
    X_current[0, -1, 1] = current_price

# Inverse Scaling
predicted_prices = scaler.inverse_transform(np.concatenate((X_current[0, -num_days:, 1].reshape(-1, 1), np.array(current_prices).reshape(-1, 1)), axis=1))[:, 1]

# Generate Dates for the Previous Lookback Days and Next Future Days
start_date = data.index[-lookback]
end_date = data.index[-1]
previous_dates = pd.date_range(start=start_date, end=end_date, freq='D')
future_dates = pd.date_range(start=end_date + pd.DateOffset(1), periods=num_days, freq='D')

def make_decision(predicted_prices):
    """
    Function to make buy/sell decisions based on the predicted prices.
    """
    deltas = [np.sign(predicted_prices[i+1] - predicted_prices[i]) for i in range(len(predicted_prices) - 1)]
    delta_changes = [deltas[i+1] - deltas[i] for i in range(len(deltas) - 1)]

    decisions = []
    for i in range(len(delta_changes)):
        if delta_changes[i] == -2:
            decisions.append('Buy')
        elif delta_changes[i] == 2:
            decisions.append('Sell')
        elif delta_changes[i] == 0:
            decisions.append('Hold')

    return decisions

# Make buy/sell decisions based on predicted prices
decisions = make_decision(predicted_prices)

# Print the decisions
for i, decision in enumerate(decisions):
    print(f"Day {i+1}: {decision}")


# Evaluate Model Performance with RMSE
y_pred = model.predict(X_test)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Price vs Time')
plt.legend()
plt.show()

# Convert the predicted prices back to their original scale.
# We only have one column, so scaler will give an error as it expects a 2D array.
# So we can just duplicate the predicted prices to create a 2D array.
y_pred = scaler.inverse_transform(np.concatenate((X_test[:, -1, 1].reshape(-1, 1), y_pred), axis=1))[:, 1]

y_test = scaler.inverse_transform(np.concatenate((X_test[:, -1, 1].reshape(-1, 1), y_test.reshape(-1, 1)), axis=1))[:, 1]

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")


# Plot the Graph for Previous Lookback Days and Future Predictions
plt.figure(figsize=(10, 6))
plt.plot(previous_dates[-lookback:], data[-lookback:]['Adj Close'].values, label='Previous Lookback Days (Adj Close)')
plt.plot(future_dates, predicted_prices, label='Predicted Future Days (Adj Close)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction for Previous Lookback Days and Future Predictions')
plt.legend()
plt.xticks(rotation=45)

# Connect the two plots with a line
plt.plot([previous_dates[-1], future_dates[0]], [data['Adj Close'].iloc[-1], predicted_prices[0]], linestyle='dashed', color='red')

plt.show()
