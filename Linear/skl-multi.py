#!/usr/bin/env python
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import datasets
from util.data_multi import load_data_multi
import matplotlib.pyplot as plt
import numpy as np
import time 

sc = StandardScaler() 

#calculate the time
start_time = time.time() 

# Load data and normalize
train_X, test_X, train_Y, test_Y = load_data_multi('iris.data')
#train_Y = np.reshape(train_Y, [len(train_Y), 1])

#test_Y = np.reshape(test_Y, [len(test_Y), 1])

#fit the model
regr = linear_model.LinearRegression()
regr.fit(train_X, train_Y)

pred = regr.predict(test_X)

#print the result
print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(test_Y, pred))
print('R2_score: %.2f' % r2_score(test_Y, pred))

