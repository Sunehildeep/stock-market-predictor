import yfinance as yf
from hyperparameters import past_years
from pytrends import dailydata
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ticker = 'TTWO'
msft = yf.Ticker(ticker)
company_name = msft.info['shortName']

t_now = datetime.now()
t_prev = t_now - timedelta(days=past_years * 365)

# trends = dailydata.get_daily_data(company_name, t_prev.year, t_prev.month, t_now.year, t_now.month)

yahoo_data = yf.download(ticker, start=t_prev, end=t_now, progress=False)
#print length
'''
Plot - Just for visualization
'''
# Plot the stock price


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
stock_prices = yahoo_data['Adj Close'].values
#Get stock_dates which is the index of the dataframe
stock_dates = yahoo_data.index.values
#Convert them to datetime objects
stock_dates = [datetime.utcfromtimestamp(date.astype('O')/1e9) for date in stock_dates]

#Add another feature to stock_prices which is the google trends data
stock_prices = np.column_stack((stock_prices, yahoo_data['Low'].values))

# Get data till 6 months before from the datetime object of stock_dates
date_6_months_before = stock_dates[-1] - timedelta(days=180)

# dataframe = pd.DataFrame({'Date':stock_dates, 'Stock Price':stock_prices})
