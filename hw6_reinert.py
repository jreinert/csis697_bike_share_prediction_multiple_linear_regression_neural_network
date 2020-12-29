# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 20:06:13 2020

@author: reine
"""
# Functions
def adj_r2(r2, n, p):
    return 1 - (((1-r2)*(n-1))/(n-p-1))

def RMSLE(y_pred, y_true):
    sum = 0.0
    for i in range (0, len(y_true)):
        log_a = math.log(y_true[i] + 1)
        log_p = math.log(y_pred[i] + 1)
        diff = (log_p - log_a)**2
        sum = sum + diff
    return math.sqrt(sum/len(y_test))
    

# Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

# Import the dataset
df = pd.read_csv('bikeshare.csv')
df = df.drop(['instant', 'dteday'], axis = 1)

# Check for null values
print('Checking for null values...\n')
print(df.isnull().sum(axis=0))
print(df.describe())

feats = df.iloc[:, 0:-1]
dep_var = df.iloc[:, -1]

# Plot scatterplots
print("Printing scatterplots...")
for f in feats:
    print(f)
    plt.figure()
    plt.xlabel(f)
    plt.ylabel('cnt')
    plt.scatter(df[f], dep_var)

# Set pandas # of columns to display to 500 for better reading AND then print correlation coefficient matrix
print("\nCorrelation Coefficient Matrix")
pd.set_option('display.max_columns', 500)
print(df.corr())

# Autocorr()
df['cnt'] = df['cnt'].astype(float)
plt.acorr(df['cnt'])

# Drop features due to linearity or multicollinearity
df = df.drop(['season', 'weekday', 'atemp', 'hum', 'casual', 'registered'], axis = 1) # <-- Added 'registered' to dropped features

df['cnt'] = np.log(df['cnt'])

# Accounting for autocorrelation on 'cnt'
t_1 = df['cnt'].shift(+1).to_frame()
t_1.columns = ['t-1']

t_2 = df['cnt'].shift(+2).to_frame()
t_2.columns = ['t-2']

t_3 = df['cnt'].shift(+3).to_frame()
t_3.columns = ['t-3']

df = pd.concat([df, t_1, t_2, t_3], axis = 1)
df = df.dropna()
df.head()

# Creating dummy variables for categorical features
df['weathersit'] = df['weathersit'].astype('category') 
df['mnth'] = df['mnth'].astype('category')
df['hr'] = df['hr'].astype('category')
df['holiday'] = df['holiday'].astype('category')
df['workingday'] = df['workingday'].astype('category')
df['yr'] = df['yr'].astype('category')

df = pd.get_dummies(df, drop_first = True)

# Train and test splits
y = df[['cnt']]
X = df.drop(['cnt'], axis = 1)

train_size = int(0.7 * len(X))
X_train = X.values[0 : train_size]
X_test = X.values[train_size : len(X)]
y_train = y.values[0 : train_size]
y_test = y.values[train_size : len(y)]


# Vars for Adj. r2 calculation
n = len(X_test)
p = len(X.columns)


""" MULTIPLE LINEAR REGRESSION """
print("\n--- MULTIPLE LINEAR REGRESSION ---")

# Train the model
mlr = LinearRegression()
mlr.fit(X_train, y_train)

# Predict the results and show the regression line
y_predict = mlr.predict(X_test)
mlr_rmse = math.sqrt(mean_squared_error(y_test, y_predict))
r_squared = mlr.score(X_test, y_test)
adjusted_rsquared = adj_r2(r_squared, n, p)

# RMSLE
y_test_exp = []
y_predict_exp = []

for i in range(0, len(y_test)):
    y_test_exp.append(math.exp(y_test[i]))
    y_predict_exp.append(math.exp(y_predict[i]))

# Print values
print('slope: ', mlr.coef_)
print('intercept: ', mlr.intercept_)
print('r squared train set: ', mlr.score(X_train, y_train))
print('r squared test set: ', r_squared)
print('adjusted r squared: ', adjusted_rsquared)
print('RMSE: ', mlr_rmse)
print('RMSLE', RMSLE(y_predict_exp, y_test_exp))

""" ARTIFICIAL NEURAL NETWORK """
print("\n--- ARTIFICIAL NEURAL NETWORK ---")

# Build the Model
input_dims = len(X.columns)
num_neurons = 10
total_num_hidden_layers = 3
num_epochs = 100
num_batchs = 5
act_func = 'relu'

model = Sequential()
model.add(Dense(num_neurons, input_dim=input_dims, activation=act_func))

i = 0
while i < total_num_hidden_layers-1:
    model.add(Dense(units = num_neurons, kernel_initializer = 'uniform', activation = act_func))
    i += 1

model.add(Dense(1, activation='linear'))

#Train the Model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batchs, verbose=1)

#Predict the Purchase Amounts
y_predict = model.predict(X_test)

print(f'\nInput Dimensions: {input_dims}')
print(f'Number of Neurons per Hidden Layer: {num_neurons}')
print(f'Total Number of Hidden Layers: {total_num_hidden_layers}')
print(f'Number of Epochs: {num_epochs}')
print(f'Batch Size: {num_batchs}')

# CALCULATE R2 VALUE
r_squared = r2_score(y_test, y_predict)
adjusted_rsquared = adj_r2(r_squared, n, p)
print('r squared: ', r_squared)
print('adjusted r squared', adjusted_rsquared)

# RMSLE
del y_test_exp[:]
del y_predict_exp[:]

for i in range(0, len(y_test)):
    y_test_exp.append(math.exp(y_test[i]))
    y_predict_exp.append(math.exp(y_predict[i]))
    
print('RMSLE', RMSLE(y_predict_exp, y_test_exp))

""" ARTIFICIAL NEURAL NETWORK """
print("\n--- ARTIFICIAL NEURAL NETWORK ---")

# Build the Model
input_dims = len(X.columns)
num_neurons = 20
total_num_hidden_layers = 5
num_epochs = 50
num_batchs = 10
act_func = 'relu'


model = Sequential()
model.add(Dense(num_neurons, input_dim=input_dims, activation=act_func))

i = 0
while i < total_num_hidden_layers-1:
    model.add(Dense(units = num_neurons, kernel_initializer = 'uniform', activation = act_func))
    i += 1

model.add(Dense(1, activation='linear'))

#Train the Model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batchs, verbose=1)

#Predict the Purchase Amounts
y_predict = model.predict(X_test)

print(f'\nInput Dimensions: {input_dims}')
print(f'Number of Neurons per Hidden Layer: {num_neurons}')
print(f'Total Number of Hidden Layers: {total_num_hidden_layers}')
print(f'Number of Epochs: {num_epochs}')
print(f'Batch Size: {num_batchs}')

# CALCULATE R2 VALUE
r_squared = r2_score(y_test, y_predict)
adjusted_rsquared = adj_r2(r_squared, n, p)
print('r squared: ', r_squared)
print('adjusted r squared', adjusted_rsquared)

# RMSLE
del y_test_exp[:]
del y_predict_exp[:]

for i in range(0, len(y_test)):
    y_test_exp.append(math.exp(y_test[i]))
    y_predict_exp.append(math.exp(y_predict[i]))
    
print('RMSLE', RMSLE(y_predict_exp, y_test_exp))

