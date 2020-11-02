# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:53:06 2020

https://www.kaggle.com/ashishpatel26/lstm-demand-forecasting

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
import plotly.offline as py
py.init_notebook_mode(connected=True)
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
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD,Adadelta,Adam,RMSprop 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout



from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
import sys
from numpy import nan
import missingno as msno 

 

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
data.to_csv(r'F:/ML_Project_April_2020/SD_Sales_Predict_ML_Projects/Sales_Imputed_Datasets/aiml_sales_day_jan20_volume.csv', index=False) 
print('\nImputed File bf1_dataset_imp.csv is Exported to location:F:/ML_Project_April_2020/SD_Sales_Predict_ML_Projects/Sales_Imputed_Datasets')

## Write imputed data set to loaction: F:/BF_SI_Prod_May_2020/ML_Model_Data/Imputed_Datasets/
# =============================================================================
# 
# # =============================================================================
# sys.stdout=open("F:/BF_SI_Prod_May_2020/ML_Model_Data/Imputed_Datasets/aiml_ppc_bf1_data_mod_imp1.csv","w")
# sys.stdout=open("F:/BF_SI_Prod_May_2020/ML_Model_Data/Imputed_Datasets/aiml_ppc_bf1_data_mod_imp1.html","w")
# print('Imputation Performed on BF1 Dataset and Exported') 
# sys.stdout.close
# 
# X = np.array(data.drop(['ANALYSIS_DATE','SI'],axis=1))
# Y = np.array(data['SI'])
# 
# =============================================================================


# data['dayofyear']=(data['dteday']-data['dteday'].apply(lambda x: date(x.year,1,1)).astype('datetime64[ns]')).apply(lambda x: x.days)

offset = int(len(data)*0.8)

# X = np.array(data.drop(columns =(['BILLING_DATE','SALES_VALUE']), axis=1, inplace=True))

X = np.array(data.drop(['BILLING_DATE','SALES_VOLUME'],axis=1))

# Test selection

# Z = np.array(data['BILLING_DATE'])
# try:
Y = np.array(data['SALES_VOLUME'])
# except KeyError:
   # print('No KeyError: SALES_VALUE')

X_train, X_test = X[:offset], X[offset:]
Y_train, Y_test = Y[:offset], Y[offset:]

# Code for RandomForestRegressor

RF = RandomForestRegressor()

RF.fit(X_train,Y_train)

print('\nMean Squared Error:')

print(mean_squared_error(Y_test,RF.predict(X_test)))

print('\nMean Squared Error:Median')

print(mean_squared_error(Y_test,np.median(Y_train)*np.ones(len(Y_test))))

print('\nCheck predictions with ground truth:')

np.vstack((RF.predict(X_test),Y_test)).T

print('\nTest Set Expected Distribution:')

## Show Graph in Pop-up window with matplotlib

# %matplotlib qt

fig1 = plt.figure()
fig1.suptitle('Test Set Expected Distribution')


pd.DataFrame(Y_test).plot(kind='hist',bins=20)

# pd.DataFrame(Y_test).plot(kind='hist',bins=20).suptitle('Test Set Expected Distribution')

print('\nTest Set Predicted Distribution:')


# =============================================================================
# fig = pyplot.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# pyplot.boxplot(results)
# ax.set_xticklabels(names)
# pyplot.show()
# 
# =============================================================================

# %matplotlib qt

fig2 = plt.figure()
fig2.suptitle('Test Set Predicted Distribution')

pd.DataFrame(RF.predict(X_test)).plot(kind='hist',bins=20)

print('\nTest Set Error Distribution:')

# %matplotlib qt

fig3 = plt.figure()
fig3.suptitle('Test Set Error Distribution')


pd.DataFrame(Y_test-RF.predict(X_test)).plot(kind='hist',bins=20)

print('\nDump ML Sales Model to a pickle object:')

with open("F:/ML_Project_April_2020/SD_Sales_Predict_ML_Projects/Model_Files/sales_vsp_day_jan20_volume.pickle", 'wb') as handle:
    pickle.dump(RF, handle, protocol=pickle.HIGHEST_PROTOCOL)


print('\nSales VSP Dataset Used: F:/ML_Project_April_2020/SD_Sales_Predict_ML_Projects/aiml_sales_day_jan20_volume.csv')

print('\nSaved sales_vsp_day_jan20.pickle model successfully to location: F:/ML_Project_April_2020/SD_Sales_Predict_ML_Projects/Model_Files/sales_vsp_day_jan20_volume.pickle')

