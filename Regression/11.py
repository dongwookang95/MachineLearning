import pandas as pd
import quandl
import math , datetime
import numpy as np
import os
#Scaling data 
#To create out training and testing sample. 
#nice way to split up data, shuffle -> no need to find biased sample
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

quandl.ApiConfig.api_key = os.environ.get('API_KEY')
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
#the high percentage of a day with features
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close']) / df['Adj. Close'] * 100.0
#Percentage change of a day 
df['PCT_CHG'] = (df['Adj. Close']-df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_CHG','Adj. Volume']]

#fill non-available or not a number. To make sure that i bring all the data 
#that i want to get
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

#when you are doing regression, it is kind of formula
forecast_out = int(math.ceil(0.01*len(df)))


#Shifting the columns negatively. A column will be shifted up 
#Each row, the label column will be adjusted clase price 10 days
#into the future
df['label'] = df[forecast_col].shift(-forecast_out)


#row
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
#column
Y = np.array(df['label'])
Y = np.array(df['label'])

#whether the table is correct or not
# print(len(X), len(Y))

#shuffle the date up. 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#as many jobs as possible
clf = LinearRegression(n_jobs = -1)

#train our data
clf.fit(X_train, Y_train)
#test our data, to make sure not to expect the value
#to do accurate work for each of them
clf.score(X_test, Y_test)
accuracy = clf.score(X_test, Y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('price')
plt.show()