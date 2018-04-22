import csv
import urllib
import re
import numpy as np
import pandas as pd
import time, math
from pandas_datareader import data as web
import tensorflow as tf
import datetime
from pandas_datareader._utils import RemoteDataError
from datetime import date
import random
from datetime import timedelta
#import fix_yahoo_finance
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Activation
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import matplotlib.pyplot as plt

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""


config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True #allocate dynamically
session = tf.Session(config=config)

#------------------ Data Collection ------------------

def lstm_data(df, start, end, tick, i):
    print ("lstm")
    tickers = df.Ticker.unique()
    data = pd.DataFrame()
    for ticker in tickers:
        stock = web.get_data_yahoo(ticker, start, end)
        #stock = web.DataReader(ticker,'iex', start, end)
        stock = pd.DataFrame(stock)
        stock = (stock.pct_change()) * 100
        news = df[(df['Ticker'] == ticker)]
        news = news.set_index('Date').join(stock)
        news = news[['Ticker', 'Title', 'Article', 'Close']]
        news = news.dropna()
        data = data.append(news)
    if i == 0:
        data = data[~(data['Ticker'] == tick)]
    if i == 1:
        data = data[(data['Ticker'] == tick)]
    #print(data)
    value = []
    for i in range(len(data['Close'])):
        if data['Close'][i] <= -2:
            #data['Close'][i] = -2
            value.append(-1)
        #elif data['Close'][i] <= -0.5 and data['Close'][i] > -2:
             #data['Close'][i] = -1
             #value.append(-1)
        elif data['Close'][i] <= 2 and data['Close'][i] > -2:
            #data['Close'][i] = 0
            value.append(0)
        #elif data['Close'][i] <= 2 and data['Close'][i] > 0.5:
            # data['Close'][i] = 1
            #value.append(1)
        elif data['Close'][i] > 2:
            # data['Close'][i] = 2
            value.append(1)
    data['Value'] = value
    print (data)
    return (data['Title'], data['Value'])


def stock_data():
    print ("stocks")
    df = pd.read_csv('StockValues.csv', index_col='date', parse_dates=True)
    df = df.pct_change()
    x = df.corr().AMD.abs() ############ TICKER ###################
    values = x.nlargest(5).values
    print(values)
    tickers = x.nlargest(5).index.values
    market_tickers = ['^IXIC', '^DJI', '^GSPC']
    tickers = np.append(tickers, market_tickers)
    #tickers = market_tickers
    print (tickers)
    i = 0

#    for value in values:
#        if value < 0.55:
#            break
#        i=i+1

#    tickers = tickers[:i]
#    print (tickers)

    delay = -1
    start = datetime.datetime.now() - datetime.timedelta(days=365 * 6)
    # start = datetime.datetime.now() - datetime.timedelta(days = 3650)
    # start = datetime.datetime.now() - datetime.timedelta(days = 3650/2)
    end = datetime.datetime.now()
    i = 0
    for ticker in tickers:
        time.sleep(5)
        try:
            if i == 0:
                print (ticker)
                stock = web.get_data_yahoo(ticker, start, end)
                #stock = web.DataReader(ticker, 'iex', start, end)
                stock = pd.DataFrame(stock)
                if stock.empty:
                    continue
                else:
                    data = stock[['Open', 'High', 'Low', 'Adj Close']]
                    i = i + 1

            else:
                print(ticker)
                stock = web.get_data_yahoo(ticker, start, end)
                #stock = web.DataReader(ticker, 'iex', start, end)
                stock = pd.DataFrame(stock)
                if stock.empty:
                    continue
                else:
                    stock = stock[['Adj Close']]
                    stock.rename(columns={'Adj Close': ticker}, inplace=True)
                    # data = stock.set_index('Date').join(data)
                    data = stock.join(data)

        except IndexError:
            print("Index Error for ticker '%s'" % ticker)
            continue
        except RemoteDataError:
            print("No information for ticker '%s'" % ticker)
            continue

    #print(data)
    data = data.pct_change()
#    for columns in data:
#        print (columns)
    for columns in data:
        if columns != 'Adj Close':
            data[columns] = data[columns].shift(delay)


    #print(data)
    data = data.dropna(how='any')
    print(data)
    return (data)

#------------------ Data Pre-Processing ---------------

def lstm_text(x_train, y_train, x_test, y_test):
    print ("Pre-processing")
    num_labels = len(np.unique(y_train))
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    stemmer = SnowballStemmer('english')

    print("pre-processing train docs...")
    processed_docs_train = []
    for doc in x_train:
        tokens = word_tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        stemmed = [stemmer.stem(word) for word in filtered]
        processed_docs_train.append(stemmed)

    print("pre-processing test docs...")
    processed_docs_test = []
    for doc in x_test:
        tokens = word_tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        stemmed = [stemmer.stem(word) for word in filtered]
        processed_docs_test.append(stemmed)

    processed_docs_all = np.concatenate((processed_docs_train, processed_docs_test), axis=0)

    dictionary = corpora.Dictionary(processed_docs_all)
    print("Dictionary: ", dictionary)
    dictionary_size = len(dictionary.keys())
    print("dictionary size: ", dictionary_size)

    print("converting to token ids...")
    word_id_train, word_id_len = [], []
    for doc in processed_docs_train:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_train.append(word_ids)
        word_id_len.append(len(word_ids))

    # print("word_id_train: ",word_id_train)

    word_id_test, word_ids = [], []
    for doc in processed_docs_test:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_test.append(word_ids)
        word_id_len.append(len(word_ids))
    # print("word_id_test :",word_id_test)

    seq_len = np.round((np.mean(word_id_len) + 2 * np.std(word_id_len))).astype(int)
    print ("seq length: ", seq_len)

    # pad sequences
    word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen=seq_len)
#    print("word_id_train pad: ", word_id_train)
#    print("word_id_train shape: ", word_id_train.shape)
    word_id_test = sequence.pad_sequences(np.array(word_id_test), maxlen=seq_len)
#    print("word_id_test pad: ", word_id_test)
#    print("word_id_test shape: ", word_id_test.shape)
    y_train_enc = np_utils.to_categorical(y_train, num_labels)
#    print("y_train_enc : ", y_train_enc)
#    print("y_train_enc shape : ", y_train_enc.shape)
    y_test_enc = np_utils.to_categorical(y_test, num_labels)
#    print("y_train_enc : ", y_test_enc)
#    print("y_train_enc shape : ", y_test_enc.shape)
    trainScore, trainAcc, testScore, testAcc, test_pred = lstm_model(dictionary_size, num_labels, word_id_train,
                                                          y_train_enc, word_id_test, y_test_enc)

    output = pd.DataFrame({'Pred': test_pred, 'Actual': y_test})
    output = output.replace(2, -1)
    #output = output.replace(1, 2)
    print(output)
    output = output.groupby(output.index).mean()
    print (output)
    return (output['Pred'])


def pre_data(stock):
    print ("nn")
    #amount_of_features = len(stock.columns)
    #data = stock.as_matrix()  # pd.DataFrame(stock)
    #data = data[1:, :]
    #row = round(0.7 * data.shape[0])
    row = round(0.7 * len(data['Open']))
    X = data.drop('Adj Close', axis=1)
    Y = data['Adj Close']
    #train = data[:int(row), :]
    x_train, x_test = X.iloc[:row], X.iloc[row:]
    y_train, y_test = Y.iloc[:row], Y.iloc[row:]
    x_train = x_train.as_matrix()
    return [x_train, y_train, x_test, y_test]


#----------------------- Models ------------------------

def lstm_model(dictionary_size, num_labels, word_id_train,
                      y_train_enc, word_id_test, y_test_enc):
    print ("lstm model")
    print ("fitting LSTM ...")
    model = Sequential()
    model.add(Embedding(dictionary_size, 128, dropout=0.3))
    model.add(LSTM(128, dropout_W=0.3, dropout_U=0.3))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    adam = optimizers.Adam(lr=0.35, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.15)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(word_id_train, y_train_enc, nb_epoch=10, batch_size=512, verbose=1)

    test_pred = model.predict_classes(word_id_test)
    trainScore, trainAcc = model.evaluate(word_id_train, y_train_enc, verbose=0)
    print('\nTrain Score: ', (trainAcc * 100), '%')
    testScore, testAcc = model.evaluate(word_id_test, y_test_enc, verbose=0)
    print('\nTest Score: ', (testAcc * 100), '%')
    return (trainScore, trainAcc, testScore, testAcc, test_pred)

def nn_model(layer):

    print("nn model")
    d = 0.1
    model = Sequential()
    model.add(Dense(128, input_dim=len(X[1, :]), kernel_initializer='uniform', activation='tanh'))
    model.add(Dropout(d))
    model.add(Dense(64, kernel_initializer='uniform', activation='tanh'))
    model.add(Dropout(d))
    model.add(Dense(16, kernel_initializer='uniform', activation='tanh'))
    model.add(Dropout(d))
    model.add(Dense(1, kernel_initializer='uniform', activation='tanh'))
    adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return model

def output(model, X, Y, Xt, Yt):

    model.fit(X, Y, batch_size=512, epochs=1000, validation_split=0.25, verbose=0)

    trainScore, trainAcc = model.evaluate(X, Y, verbose=0)
    # print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0]), trainAcc[1]))
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))

    testScore, testAcc = model.evaluate(Xt, Yt, verbose=0)
    # print('Test Score: %.2f MSE (%.2f RMSE), Test Accuracy %.2f' % (testScore[0], math.sqrt(testScore[0]), testAcc[1]))
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

    # print(x_test[-1])
    diff = []
    ratio = []
    p = model.predict(Xt)
    #print (p)
    for u in range(len(Yt)):
        pr = p[u][0]
        ratio.append((Yt[u] / pr) - 1)
        diff.append(abs(Yt[u] - pr))
        #print(u, Yt[u], pr, (Yt[u] / pr) - 1, abs(Yt[u] - pr))

    # print(p)
    # print(y_test)
    Yt = pd.DataFrame(Yt)
    Yt['Pred'] = p

    return (Yt)

    """
    plt2.plot(p, color='red', label='prediction')
    plt2.plot(Yt, color='blue', label='y_test')
    plt2.legend(loc='upper left')
    plt2.grid(color='k', linestyle='-')
    plt2.show()
    if (i==0):
        plt2.savefig('Main')
    elif (i == 1):
        plt2.savefig('Deep')
    """



if __name__=='__main__':
    print ("main")
    ticker = 'AMD'
    nondata = stock_data()
    #info = pd.read_csv('news_data1.csv', encoding="cp1252",index_col='Date') #, parse_dates=True
    info = pd.read_csv('news_data1.csv', encoding="utf8", index_col='Date')  # , parse_dates=True
#    print(info)
#    x_train, y_train = lstm_data(df, (date.today() - datetime.timedelta(days=1500)),
#                                   date.today(), ticker, 0)
#    x_test, y_test = lstm_data(df, (date.today() - datetime.timedelta(days=1500)),
#                                   date.today(), ticker, 1)

#    df = info[~(info['Ticker'] == ticker)]
#    print (df)
#    x_train, y_train = df['Article'], df['Value']



    df = info[(info['Ticker'] == ticker)]

#    print(type(df))
#    print(df)
#    print(df['Close'][1])
    
#    new_value = []
#    for i in range(len(df['Close'])):
#        if df['Close'][i] <= -1.3:
            #data['Close'][i] = -2
#            new_value.append(-1)
        #elif data['Close'][i] <= -0.5 and data['Close'][i] > -2:
             #data['Close'][i] = -1
             #value.append(-1)
#        elif df['Close'][i] <= 1.3 and df['Close'][i] > -1.3:
            #data['Close'][i] = 0
#            new_value.append(0)
        #elif data['Close'][i] <= 2 and data['Close'][i] > 0.5:
            # data['Close'][i] = 1
            #value.append(1)
#        elif df['Close'][i] > 1.3:
            # data['Close'][i] = 2
#            new_value.append(1)

#    print(df)

    
    X,Y = df['Title'],df ['Value']
    row = round(0.8 * X.shape[0])

    x_train, x_test = X.iloc[0:row], X.iloc[row:]
    y_train, y_test = Y.iloc[0:row], Y.iloc[row:]

    lstm_pred = lstm_text(x_train, y_train, x_test, y_test)
    #print (lstm_pred)
    lstm_pred = pd.DataFrame(lstm_pred)
    #print (lstm_pred)


    data = lstm_pred.join(nondata)
    data = data.dropna(how='any')
    print ("Complete Data: \n", data)

#-------- LSTM and Correlation ---------------

    X,Y, Xt, Yt = pre_data(data)
    model = nn_model(X)
    output_all = output(model, X, Y, Xt, Yt)
    print("output: \n", output_all)

# -------- Correlation ---------------

    for columns in data:
        if columns == 'Pred':
            data1 = data.drop(columns, axis=1, inplace=False)
    print("Correlation Data: \n", data1)
    X, Y, Xt, Yt = pre_data(data1)
    model1 = nn_model(X)
    output_corr = output(model1, X, Y, Xt, Yt)
    print("output_corr: \n", output_corr)

# -------- LSTM  ---------------
#    data = data[:-4,:]
    for columns in data:
        if columns != 'Pred' and columns != 'High' and columns != 'Low' and columns != 'Adj Close' and columns != 'Open':
            data2 = data.drop(columns, axis=1, inplace=False)
    print("LSTM Data: \n", data2)
    X, Y, Xt, Yt = pre_data(data2)
    model2 = nn_model(X)
    output_lstm = output(model2, X, Y, Xt, Yt)
    print("output_lstm: \n", output_lstm)

    # -------- No LSTM or Correlation ---------------

    for columns in data1:
        if columns != 'Open' and columns != 'High' and columns != 'Low' and columns != 'Adj Close':
            data1 = data1.drop(columns, axis=1, inplace=False)
    print("Final Data: \n", data1)
    X, Y, Xt, Yt = pre_data(data1)
    model3 = nn_model(X)
    output_none = output(model3, X, Y, Xt, Yt)
    print("output_none: \n", output_none)


    plt.subplot(411)
    plt.plot(output_all['Pred'], color='red', label='prediction')
    plt.plot(output_all['Adj Close'], color='blue', label='y_test')
    plt.legend(loc='upper left')
    plt.grid(color='k', linestyle='-')
    plt.title('LSTM And Correlated Stocks')

    plt.subplot(412)
    plt.plot(output_corr['Pred'], color='red', label='prediction')
    plt.plot(output_corr['Adj Close'], color='blue', label='y_test')
    plt.legend(loc='upper left')
    plt.grid(color='k', linestyle='-')
    plt.title('No-Lstm')

    plt.subplot(413)
    plt.plot(output_lstm['Pred'], color='red', label='prediction')
    plt.plot(output_lstm['Adj Close'], color='blue', label='y_test')
    plt.legend(loc='upper left')
    plt.grid(color='k', linestyle='-')
    plt.title('No-Corr')

    plt.subplot(414)
    plt.plot(output_none['Pred'], color='red', label='prediction')
    plt.plot(output_none['Adj Close'], color='blue', label='y_test')
    plt.legend(loc='upper left')
    plt.grid(color='k', linestyle='-')
    plt.title('No LSTM or Correlation')
    plt.show()



