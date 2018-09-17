

import numpy as np
import pandas as pd
import time, math
from pandas_datareader import data as web
import tensorflow as tf
import datetime
from pandas_datareader._utils import RemoteDataError
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Activation
from keras.preprocessing import sequence
from keras.utils import np_utils
import matplotlib.pyplot as plt

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True  # allocate dynamically
session = tf.Session(config=config)


# ------------------ Data Selection ------------------

def data(ticker):
    print("Stocks")
    if ticker == 'GOOGL':
        data = pd.read_csv('Google.csv', index_col='Date', parse_dates=True)
        return data

    elif ticker == 'AMZN':
        data = pd.read_csv('Amazon.csv', index_col='Date', parse_dates=True)
        return data

    elif ticker == 'AMD':
        data = pd.read_csv('AMD.csv', index_col='Date', parse_dates=True)
        return data

    elif ticker == 'AAPL':
        data = pd.read_csv('Apple.csv', index_col='Date', parse_dates=True)
        return data

    elif ticker == 'MSFT':
        data = pd.read_csv('Microsoft.csv', index_col='Date', parse_dates=True)
        return data

    elif ticker == 'NFLX':
        data = pd.read_csv('Netflix.csv', index_col='Date', parse_dates=True)
        return data

    elif ticker == 'NVDA':
        data = pd.read_csv('Nvidia.csv', index_col='Date', parse_dates=True)
        return data


# ------------------ Data Pre-Processing ---------------

def lstm_text(x_train, y_train, x_test, y_test):
    print("Pre-processing")
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


    word_id_test, word_ids = [], []
    for doc in processed_docs_test:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_test.append(word_ids)
        word_id_len.append(len(word_ids))

    seq_len = 50
    word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen=seq_len)
    word_id_test = sequence.pad_sequences(np.array(word_id_test), maxlen=seq_len)
    y_train_enc = np_utils.to_categorical(y_train, num_labels)
    y_test_enc = np_utils.to_categorical(y_test, num_labels)
    trainScore, trainAcc, testScore, testAcc, test_pred = lstm_model(dictionary_size, num_labels, word_id_train,
                                                                     y_train_enc, word_id_test, y_test_enc)

    output = pd.DataFrame({'Pred': test_pred, 'Actual': y_test}).replace(2, -1)
    output = output.groupby(output.index).mean()
    print(output)
    return (output['Pred'])


def pre_data(data):
    row = round(0.7 * len(data['Open']))
    X = data.drop('Adj Close Output', axis=1)
    Y = data['Adj Close Output']
    x_train, x_test = X.iloc[:row], X.iloc[row:]
    y_train, y_test = Y.iloc[:row], Y.iloc[row:]
    x_train = x_train.as_matrix()
    return [x_train, y_train, x_test, y_test]


def lstm_output(data):
    a = b = c = i = k = 0
    xyz = {}
    new_data = []

    idx = data.index.unique()
    idx = pd.DataFrame(idx)
    idx = idx.drop(idx.index[len(idx) - 1])

    for index in data.index:
        if index == data.index[k]:
            xyz['Actual'] = data.iloc[k, 0]
            if data.iloc[k, 1] == 0:
                c = c + 1
            elif data.iloc[k, 1] == 1:
                a = a + 1
            elif data.iloc[k, 1] == -1:
                b = b + 1
            k = k + 1
        if (k >= len(data.index)):
            break
        elif index != data.index[k]:
            if a == b or a < 0.05 * c or b < 0.05 * c:
                xyz['Pred'] = 0
            if a > b and a > 0.05 * c:
                xyz['Pred'] = 0.5
            if a < b and b > 0.05 * c:
                xyz['Pred'] = -0.5
            i = i + 1
            a = 0
            b = 0
            c = 0
            new_data.append(xyz)
        xyz = {}

    new_data = pd.DataFrame(new_data)
    new_data = idx.join(new_data)
    new_data = new_data.set_index('Date')
    new_data.to_csv('trail2.csv')
    return new_data


# ----------------------- Models ------------------------

def lstm_model(dictionary_size, num_labels, word_id_train,
               y_train_enc, word_id_test, y_test_enc):
    print("lstm model")
    print("fitting LSTM ...")
    model = Sequential()
    model.add(Embedding(dictionary_size, 128, dropout=0.3))
    model.add(LSTM(128, dropout_W=0.3, dropout_U=0.3))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    adam = optimizers.Adam(lr=0.35, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.05)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(word_id_train, y_train_enc, nb_epoch=10, batch_size=512, verbose=1)

    test_pred = model.predict_classes(word_id_test)
    trainScore, trainAcc = model.evaluate(word_id_train, y_train_enc, verbose=0)
    print('\nTrain Score: ', (trainAcc * 100), '%')
    testScore, testAcc = model.evaluate(word_id_test, y_test_enc, verbose=0)
    print('\nTest Score: ', (testAcc * 100), '%')
    return (trainScore, trainAcc, testScore, testAcc, test_pred)


def nn_model(X):
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
    model.compile(loss='mse', optimizer=adam, metrics=['mse'])
    return model


def output(model, X, Y, Xt, Yt):
    model.fit(X, Y, batch_size=512, epochs=500, validation_split=0.25, verbose=0)

    trainScore, trainAcc = model.evaluate(X, Y, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))

    testScore, testAcc = model.evaluate(Xt, Yt, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


    diff = []
    ratio = []
    p = model.predict(Xt)
    for u in range(len(Yt)):
        pr = p[u][0]
        ratio.append((Yt[u] / pr) - 1)
        diff.append(abs(Yt[u] - pr))

    Yt = pd.DataFrame(Yt)
    Yt['Pred'] = p

    return (Yt)


if __name__ == '__main__':
    ticker = 'AMD'
    dataRaw = data(ticker)

    info = pd.read_csv('news_data1.csv', encoding="utf8", index_col='Date')
    df = info[(info['Ticker'] == ticker)]

    X, Y = df['Article'], df['Value']
    row = round(0.8 * X.shape[0])

    x_train, x_test = X.iloc[0:row], X.iloc[row:]
    y_train, y_test = Y.iloc[0:row], Y.iloc[row:]

    lstm_pred = lstm_text(x_train, y_train, x_test, y_test)

    lstm_pred = pd.DataFrame(lstm_pred)

    data = lstm_pred.join(dataRaw)
    data = data.dropna(how='any')


#-------- LSTM and Correlation ---------------

    print("####################   LSTM and CORRELATION DATA     ###############")
    X,Y, Xt, Yt = pre_data(data)
    model = nn_model(X)
    output_all = output(model, X, Y, Xt, Yt)

# -------- Correlation ---------------
    data1 = data
    for columns in data:
        if columns == 'Pred':
            data1 = data1.drop(columns, axis=1, inplace=False)
    print("####################   NO LSTM and CORRELATION DATA     ###############")
    X1, Y1, Xt1, Yt1 = pre_data(data1)
    model1 = nn_model(X1)
    output_corr = output(model1, X1, Y1, Xt1, Yt1)
    #print("Correlation Data: \n", data1)

# -------- LSTM  ---------------
    data2 = data
    for columns in data:
        if columns != 'Pred' and columns != 'Open' and columns != 'High' and columns != 'Low' and columns != 'Adj Close' and columns != 'Adj Close Output':
            data2 = data2.drop(columns, axis=1, inplace=False)
    print("####################   LSTM and NO CORRELATION DATA     ###############")
    X2, Y2, Xt2, Yt2 = pre_data(data2)
    model2 = nn_model(X2)
    output_lstm = output(model2, X2, Y2, Xt2, Yt2)
    #print("LSTM Data: \n", data2)


    # -------- No LSTM or Correlation ---------------
    data3 = data
    for columns in data:
        if columns != 'Open' and columns != 'High' and columns != 'Low' and columns != 'Adj Close' and columns != 'Adj Close Output':
            data3 = data3.drop(columns, axis=1, inplace=False)
    print("####################   NO LSTM and NO CORRELATION DATA     ###############")
    X3, Y3, Xt3, Yt3 = pre_data(data3)
    model3 = nn_model(X3)
    output_none = output(model3, X3, Y3, Xt3, Yt3)
    #print("Final Data: \n", data3)

    plt.subplot(411)
    plt.plot(output_all['Pred'], color='red', label='prediction')
    plt.plot(output_all['Adj Close Output'], color='blue', label='y_test')
    plt.legend(loc='upper left')
    plt.grid(color='k', linestyle='-')
    plt.title('LSTM And Correlated Stocks')
#    plt.show()


    plt.subplot(412)
    plt.plot(output_corr['Pred'], color='red', label='prediction')
    plt.plot(output_corr['Adj Close Output'], color='blue', label='y_test')
    plt.legend(loc='upper left')
    plt.grid(color='k', linestyle='-')
    plt.title('No-Lstm')
#    plt.show()


    plt.subplot(413)
    plt.plot(output_lstm['Pred'], color='red', label='prediction')
    plt.plot(output_lstm['Adj Close Output'], color='blue', label='y_test')
    plt.legend(loc='upper left')
    plt.grid(color='k', linestyle='-')
    plt.title('No-Corr')
#    plt.show()

    plt.subplot(414)
    plt.plot(output_none['Pred'], color='red', label='prediction')
    plt.plot(output_none['Adj Close Output'], color='blue', label='y_test')
    plt.legend(loc='upper left')
    plt.grid(color='k', linestyle='-')
    plt.title('No LSTM or Correlation')
    plt.show()

