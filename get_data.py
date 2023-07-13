import yfinance as yf
from hyperparameters import past_history
from pytrends import dailydata
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

ticker = 'XOM'
msft = yf.Ticker(ticker)
company_name = msft.info['shortName']

t_now = datetime.now()
t_prev = t_now - timedelta(days=past_history)

trends = dailydata.get_daily_data(company_name, t_prev.year, t_prev.month, t_now.year, t_now.month)

yahoo_data = msft.history(period='1d', start=t_prev, end=t_now)

'''
Plot - Just for visualization
'''
# Plot the stock price
# plt.figure(figsize=(10, 5))
# plt.plot(yahoo_data['Close'].values, label='Stock Price')
# plt.xlabel('Time')
# plt.ylabel('Stock Price')
# plt.title('Stock Price vs Time')
# plt.legend()
# plt.show()

# # Plot the google trends data
# plt.figure(figsize=(10, 5))
# plt.plot(trends[company_name].values, label='Google Trends')    
# plt.xlabel('Time')
# plt.ylabel('Google Trends')
# plt.title('Google Trends vs Time')
# plt.legend()
# plt.show()


# Extract the relevant data
print(yahoo_data.columns)
stock_prices = yahoo_data['Close'].values.reshape(-1, 1)
google_trends = trends[company_name].values.reshape(-1, 1)

#Scale and normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
stock_prices = scaler.fit_transform(stock_prices)
google_trends = scaler.fit_transform(google_trends)

# Combine the data
# input_data = np.concatenate((stock_prices, google_trends), axis=1)