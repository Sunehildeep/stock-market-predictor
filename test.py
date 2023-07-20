# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:56:53 2023

@author: pcgam
"""
import yfinance as yf
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define past_years
past_years = 5

tickers = ['UBER','TTWO','SHOP','PYPL','INTC','GOOG','ABNB','NVDA','MDB']
    
for ticker in tickers:
    
    t_now = datetime.now()
    t_now = t_now - timedelta(days=1)
    t_prev = t_now - timedelta(days=past_years * 365)
    
    data = yf.download(ticker, start=t_prev, end=t_now, progress=False)
    
    data = data['Adj Close']
    
    data = data.resample('D').last()
    
    data.dropna(inplace=True)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    lookback = 100
    
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    
    # Load the checkpoint
    model.load_weights('TTWO_model.h5')
    
    # Predict the future price
    future_price = model.predict(np.array([scaled_data[-lookback:]]))[0][0]
    
    # Predict Future Stock Prices
    X_current = scaled_data[-lookback:].reshape(1, -1, 1)  # Last lookback days in the available data
    current_prices = []
    
    num_days = 30 # Number of days to predict into the future
    for _ in range(num_days):
        current_price = model.predict(X_current)
        current_prices.append(current_price[0, 0])
        X_current = np.roll(X_current, -1)  # Shift the input sequence by one day
        X_current[0, -1, 0] = current_price
    
    # Inverse Scaling
    predicted_prices = scaler.inverse_transform(np.concatenate((X_current[0, -num_days:, 0].reshape(-1, 1), np.array(current_prices).reshape(-1, 1)), axis=1))[:, 1]
    
    # Generate Dates for the Previous Lookback Days and Next Future Days
    start_date = data.index[-lookback]
    end_date = data.index[-1]
    previous_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    future_dates = pd.date_range(start=end_date + pd.DateOffset(1), periods=num_days, freq='D')
           
    # Plot the Graph for Previous Lookback Days and Future Predictions
    plt.figure(figsize=(10, 6))
    plt.plot(previous_dates[-lookback:], data[-lookback:].values, label='Previous Lookback Days (Adj Close)')
    plt.plot(future_dates, predicted_prices, label='Predicted Future Days (Adj Close)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'Stock {ticker} Price Prediction for Previous Lookback Days and Future Predictions')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Connect the two plots with a line
    plt.plot([previous_dates[-1], future_dates[0]], [data.iloc[-1], predicted_prices[0]], linestyle='dashed', color='red')
    
    # Save the plot as a PNG file in {ticker}_plots folder along with .txt file containing the decisions
    # Create a folder with the ticker name if it doesn't exist
    if not os.path.exists(f'{ticker}_plots'):
        os.makedirs(f'{ticker}_plots')

    plt.savefig(f'{ticker}_plots/{ticker}_plot.png')
    plt.show()
    
    
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

    
    with open(f'{ticker}_plots/{ticker}_decisions.txt', 'w') as f:
        for i, decision in enumerate(decisions):
            f.write(f"Day {i+1}: {decision}\n")
        