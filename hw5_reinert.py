# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 20:06:13 2020

@author: reine
"""
# Functions
def adj_r2(r2, n, p):
    return 1 - (((1-r2)*(n-1))/(n-p-1))
    

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
df = df.drop(['season', 'weekday', 'atemp', 'hum', 'casual', 'registered'], axis = 1)

X = df.iloc[:, 0:-1] 
y = df.iloc[:, -1]

X['weathersit'] = X.weathersit.astype('category')
X['mnth'] = X.mnth.astype('category')
X['hr'] = X.hr.astype('category')
X['holiday'] = X.holiday.astype('category')
X['workingday'] = X.workingday.astype('category')
X['yr'] = X.yr.astype('category')

# One Hot Encoding
X = pd.get_dummies(X, drop_first = True)

# Split into Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

# Print values
print('slope: ', mlr.coef_)
print('intercept: ', mlr.intercept_)
print('r squared: ', r_squared)
print('adjusted r squared: ', adjusted_rsquared)
print('RMSE: ', mlr_rmse)

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
