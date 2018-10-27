#!/usr/bin/env python
from sklearn import linear_model
from sklearn.datasets import load_boston
from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
sc = StandardScaler() 


def normaliseData(x):
  # rescale data to lie between 0 and 1
  scale = x.max()
  return (x/scale)

(train_X, train_Y), (test_X, test_Y) = boston_housing.load_data()

train_X = normaliseData(train_X)
test_X = normaliseData(test_X)

train_Y = np.reshape(train_Y, [404, 1])
train_Y = normaliseData(train_Y)
test_Y = np.reshape(test_Y, [len(test_Y), 1])
test_Y = normaliseData(test_Y)


#fit the model
regr = linear_model.LinearRegression()
regr.fit(train_X, train_Y)

pred = regr.predict(test_X)


#print the result
print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(test_Y, pred))
print('Variance score: %.2f' % r2_score(test_Y, pred))



   