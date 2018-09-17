# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:36:05 2017

@author: hari0
"""

import numpy as np
import pandas as pd
import time
from pandas_datareader import data as web
import datetime
from pandas_datareader._utils import RemoteDataError


df = pd.read_csv('StockValues.csv', index_col='date', parse_dates=True)
df = df.pct_change()

x = df.corr().NFLX.abs()

xyz = x.nlargest(5).values
tickers = x.nlargest(5).index.values
delay = -1
print(tickers)
print(xyz)

def stock_data(tickers, delay):
    start = datetime.datetime.now() - datetime.timedelta(days=365)
    end = datetime.datetime.now()
    i = 0
    for ticker in tickers:
        time.sleep(5)
        try:
            if i == 0:
                stock = web.get_data_yahoo(ticker, start, end)
                stock = pd.DataFrame(stock)
                if stock.empty:
                    continue
                else:
                    data = stock[['Open', 'High', 'Low', 'Adj Close']]
                    i = i + 1

            else:
                stock = web.get_data_yahoo(ticker, start, end)
                stock = pd.DataFrame(stock)
                if stock.empty:
                    continue
                else:
                    stock = stock[['Adj Close']]
                    stock.rename(columns={'Adj Close': ticker}, inplace=True)
                    data = stock.join(data)

        except RemoteDataError:
            print("No information for ticker '%s'" % ticker)
            continue
        except IndexError:
            print("Index Error for ticker '%s'" % ticker)
            continue

    data = data.pct_change()
    data['Open'] = data['Open'].shift(delay)
    data['High'] = data['High'].shift(delay)
    data['Low'] = data['Low'].shift(delay)
    data = data.dropna(how='any')

    return data

stock = stock_data(tickers, delay)
print(stock)
print(type(stock))
