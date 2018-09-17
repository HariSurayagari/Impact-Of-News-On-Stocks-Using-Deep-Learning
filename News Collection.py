# -*- coding: utf-8 -*-

import pandas as pd
from datetime import date
from pandas_datareader import data as web

# AN API CALLED EVENT REGISTRY HELPS WITH GETTING THE NEWS NEEDED FOR THIS PROJECT

from eventregistry import *
er = EventRegistry(apiKey="97e10653-b4b8-422a-9dab-0848cae6965f")

names = ['Apple', 'AMD', 'Amazon', 'Nvidia', 'Microsoft', 'Google', 'Netflix']
tickers = ['AAPL', 'AMD', 'AMZN', 'NVDA',  'MSFT', 'GOOGL', 'NFLX']
text = []

# ----------- Data Collection --------------

def build_data(name, tickers, start, end):

    i = 0
    data = pd.DataFrame()
    for name in names:
        news = []
        q = QueryArticlesIter(conceptUri=er.getConceptUri(name), lang = 'eng', sourceUri=QueryItems.OR(
            ["marketwatch.com", "investopedia.com", "bloomberg.com", "fool.com", "money.cnn.com",
             "reuters.com", "stocknews.com", "thestreet.com", "nasdaq.com", "investors.com"]), dateStart= start, dateEnd=end )
        for art in q.execQuery(er, sortBy="date", articleBatchSize=200,maxItems=-1,
                               returnInfo=ReturnInfo(articleInfo=ArticleInfoFlags(originalArticle=True),
                                                     categoryInfo=CategoryInfoFlags(trendingSource="news"))):
            dic = {}
            dic['Date'] = art['date']
            dic['Name'] = name
            dic['Title'] = art['title']
            dic['Article'] = art['body']

            if i == 0:
                news.append(dic)
                i = i + 1

            elif dic['Title'] not in news[i - 1]['Title'] and dic['Article'] not in news[i - 1]['Article']:
                news.append(dic)

                i = i + 1

        news = pd.DataFrame(news)
        data = data.append(news)
        i = 0


    #print(data)
    #data.to_csv('final_data.csv')
    return [data]

def lstm_data(df, tickers, start, end):

    data = pd.DataFrame()
    for ticker in tickers:
        stock = web.get_data_yahoo(ticker, start, end)
        stock = pd.DataFrame(stock)
        stock = (stock.pct_change()) * 100
        news = df[(df['Ticker'] == ticker)]
        news = news.set_index('Date').join(stock)
        news = news[['Ticker', 'Title', 'Article', 'Close']]
        news = news.dropna()
        data = data.append(news)

    value = []
    for i in range(len(data['Close'])):
        if data['Close'][i] <= -2:
            value.append(-1)
        elif data['Close'][i] <= 2 and data['Close'][i] > -2:
            value.append(0)
        elif data['Close'][i] > 2:
            value.append(1)
    data['Value'] = value
    print(data)
    return (data)

data = build_data(tickers, (date.today() - datetime.timedelta(days=2000)),
                              date.today())

data = lstm_data(data, tickers, (date.today() - datetime.timedelta(days=2000)),
                              date.today())