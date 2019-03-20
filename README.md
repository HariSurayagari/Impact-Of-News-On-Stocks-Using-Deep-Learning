# Stock Predictions Using Deep Learning

The repository is split into 3 parts: News Collection, Correlated Stocks, Stock Predictions. 

News Collections - This part of the code is used to collect the data using an API called event registry to collect news from various financial websites over the last 6 years related to companies like AMD, NVIDIA, GOOGLE, NETFLIX, MICROSOFT, AMAZON. 

Correleated Stocks - This part is used to generate the stock data of the above companies and the most correlated stocks for each of thoses companies. This part also combines the data collected from the news and preprocesses it, getting the data ready for the final step. 

Stock Predictions - Here the processed data is put through an LSTM network. The training data is the news articles generated between 2012-2016. The network provides some positive results and this output data is combined with the stock data of the corresponding company's stock data over the last 1 year (2016-2017). This data is used on a Deep Neural Networks. 
The final output seems to give a RMSE values of about 0.01.to 0.03 on an average for the companies. 
