# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:53:06 2020

https://towardsdatascience.com/predicting-sales-611cb5a252de

Usage: Predicting Sales : Forecasting monthly sales with LSTM

Dataset: 'F:/ML_Project_April_2020/SD_Sales_Predict_ML_Projects/aiml_sales_day_jan20_volume.csv'

@author: 119540
"""
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set()
# %matplotlib inline

import chart_studio.plotly as py
import plotly.offline as pyoff
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import statsmodels.api as sm
# import xgboost as xgb
# import lightgbm as lgb
from sklearn.model_selection import train_test_split

import missingno as msno 

import warnings
# import the_module_that_warns

warnings.filterwarnings("ignore")


# from fbprophet import Prophet

## for Deep-learing:
import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD,Adadelta,Adam,RMSprop 
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.utils import np_utils
from keras.utils import np_utils
import itertools
from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers.convolutional import Conv1D
# from tensorflow.keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import KFold, cross_val_score, train_test_split


from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
import sys
from numpy import nan
import missingno as msno 

#initiate plotly
# pyoff.init_notebook_mode()

data = pd.read_csv('F:/ML_Project_April_2020/SD_Sales_Predict_ML_Projects/aiml_sales_day_jan20_volume.csv',  delimiter=',', encoding= 'unicode_escape', header=0, parse_dates=['BILLING_DATE'])

# data = pd.read_csv('F:/ML_Project_April_2020/SD_Sales_Predict_ML_Projects/aiml_sales_day_jan20.csv',  encoding='unicode_escape', sep='\s*,\s*', engine='python', header=0)

data.columns = data.columns.str.strip()

# Convert Data type: objects to float to remove ValueError: could not convert string to float

# data['SALES_OFFICE'] = data['SALES_OFFICE'].astype(float)

data['SALES_OFFICE'] = pd.to_numeric(data['SALES_OFFICE'], errors='coerce')

# data['REGION_CODE'] = data['REGION_CODE'].astype(float)

data['REGION_CODE'] = pd.to_numeric(data['REGION_CODE'], errors='coerce')
# F:/ML_Project_April_2020/SD_Sales_Predict_ML_Projects/Sales_SD_Sample_ML_Projects/Month_Sales_JasonBrownie_Dataset.csv
# , header=0, index_col=['BILLING_DATE']
print('Data Shape')
print('\n-----------------')
print(data.info)
print(data.head(10))
print('Shape:', data.shape)


print('\nAnalyzing missing Values in Dataset')
print('\n-------------------------------------')

# Visualize missing values as a matrix 
msno.matrix(data) 

# Visualize the number of missing values as a bar chart 
msno.bar(data) 

# Visualize the correlation between the number of missing values in different columns as a heatmap 
msno.heatmap(data) 

# fill missing values with mean column values
data.fillna(data.mean(), inplace=True)
# count the number of NaN values in each column
print('\nSummary on Null Values')
print('\n----------------------------')
print(data.isnull().sum())

data.head(20)
print(data.describe())
data.info()
print('\nWriting Imputed Data File:bf2_dataset_imp.csv to location F:/ML_Project_April_2020/SD_Sales_Predict_ML_Projects/Sales_Imputed_Datasets')
data.to_csv(r'F:/ML_Project_April_2020/SD_Sales_Predict_ML_Projects/Sales_Imputed_Datasets/aiml_sales_mon_jan20_volume.csv', index=False) 
print('\nImputed File bf1_dataset_imp.csv is Exported to location:F:/ML_Project_April_2020/SD_Sales_Predict_ML_Projects/Sales_Imputed_Datasets')


# Task: Predict monthly total sales. Aggregate data at the monthly level and sum up the sales column.
# Represent month in date field as its first day

data['BILLING_DATE'] = data['BILLING_DATE'].dt.year.astype('str') + '-' + data['BILLING_DATE'].dt.month.astype('str') + '-01'
data['BILLING_DATE'] = pd.to_datetime(data['BILLING_DATE'])

# Groupby date and sum the sales

data = data.groupby('BILLING_DATE').SALES_VOLUME.sum().reset_index()

#  data is now showing the aggregated sales - Repalce df_sales with data

print('\n--------------------------------------------------')
print('Month Wise Sales Volume Summary in each Year')
print('\n--------------------------------------------------')

print(data.head())
print('\n--------------------------------------------------')
print('Data Transformation')
print('\n--------------------------------------------------')

'''
Convert the data to stationary if it is not Converting from time series to supervised
for having the feature set of LSTM model Scale the data- check if the data is not stationary? see plot: 
'''

#plot monthly sales

plot_data = [
    go.Scatter(
        x=data['BILLING_DATE'],
        y=data['SALES_VOLUME'],
    )
]

plot_layout = go.Layout(
        title='VSP Monthly Sales - Volume Analysis'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
# Below code prints plot as html page in new window in browser
pyoff.plot(fig)

'''
Data is not stationary and has an increasing trend over the months.
Get the difference in sales compared to the previous month and build the model on it:
'''
#create a new dataframe to model the difference in sales volume
# df_diff = data.copy() is causing keyerror and hence changed df_diff = data.copy() to df_diff = data

# df_diff = data.copy()

df_diff = data

#add previous sales to the next row
data['prev_sales'] = df_diff['SALES_VOLUME'].shift(1)
#drop the null values and calculate the difference
df_diff = df_diff.dropna()
df_diff['diff'] = (df_diff['SALES_VOLUME'] - df_diff['prev_sales'])
    
print('\n--------------------------------------------------')
print(df_diff.head(10))
print('\n--------------------------------------------------')

#plot sales diff

plot_data = [
    go.Scatter(
        x=df_diff['BILLING_DATE'],
        y=df_diff['diff'],
    )
]

plot_layout = go.Layout(
        title='VSP Monthly Sales - Volume Differences Over Previous Month'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
# Below code prints plot as html page in new window in browser
pyoff.plot(fig)

'''
Build feature set: Need to use previous monthly sales data to predict the next ones. 
The look-back period may vary for every model. Used 9 for this model.
Create columns from lag_1 to lag_12 and assign values by using shift() method:
'''
#Create dataframe for transformation from time series to supervised

df_supervised = df_diff.drop(['prev_sales'],axis=1)
#adding lags
for inc in range(1,13):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(inc)
    #drop null values
    df_supervised = df_supervised.dropna().reset_index(drop=True)

print('\n--------------------------------------------------')
print(df_supervised.head(10))
print('\n--------------------------------------------------')

'''
Using features for prediction?
Adjusted R-squared:It tells us how good our features explain the variation in label 
(lag_1 to lag_12 for diff, in our example). - Difference values not used for model - use data
'''
# Import statsmodels.formula.api

# import statsmodels.formula.api as smf
# # Define the regression formula
# model = smf.ols(formula='diff ~ lag_1', data=df_supervised)
# # Fit the regression
# model_fit = model.fit()
# # Extract the adjusted r-squared
# regression_adj_rsq = model_fit.rsquared_adj
# print('\n--------------------------------------------------')
# print(regression_adj_rsq)
# print('\n--------------------------------------------------')
'''
Build model after scaling  data. But there is one more step before scaling. 
Split our data into train and test sets.For test set, selected the last 6 months’ sales.
'''

#import MinMaxScaler and create a new dataframe for LSTM model
from sklearn.preprocessing import MinMaxScaler

df_model = data.drop(['SALES_VOLUME','BILLING_DATE'],axis=1)
#split train and test set
train_set, test_set = df_model[0:-6].values, df_model[-6:].values

'''
As the scaler, used MinMaxScaler, which will scale each future between -1 and 1:  
'''
#apply Min Max Scaler

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)

# reshape training set

train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)

'''
Building the LSTM model:Let’s create feature and label sets from scaled datasets:Check below code
'''
X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


# Fit LSTM model:

model = Sequential()
model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=1, shuffle=False)

# Do prediction and see how the results look like:
    
y_pred = model.predict(X_test,batch_size=1)

# For multistep prediction, you need to replace X_test values with the predictions coming from t-1



