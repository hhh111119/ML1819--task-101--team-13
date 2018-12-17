#!/usr/bin/env python
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
sc = StandardScaler() 


# Rescale data to lie between 0 and 1
def normaliseData(x): 
  scale = x.max()
  return (x/scale)

#Load data
iris = datasets.load_iris()


# choose petal length and petal width as feature via chi-squared test.
X = SelectKBest(chi2, k=3).fit_transform(iris.data, iris.target)
y=iris.target
#Split the train and test data
train_X, test_X, train_Y, test_Y = train_test_split(X, y,test_size=0.3, random_state=None)

#Normalize
train_X = normaliseData(train_X)
test_X = normaliseData(test_X)
train_Y = np.reshape(train_Y, [len(train_Y), 1])
test_Y = np.reshape(test_Y, [len(test_Y), 1])

#fit the model
regr = linear_model.LinearRegression()
regr.fit(train_X, train_Y)

pred = regr.predict(test_X)

#print the result
print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(test_Y, pred))
print('R2_score: %.2f' % r2_score(test_Y, pred))

plt.plot([1,2,3], [0.93,0.95,0.96], 'ro','b-', label='Best fit line', linewidth=3)
plt.title('R²-n features')
plt.xlabel('Amount of feature')
plt.ylabel('R²')
plt.show()
