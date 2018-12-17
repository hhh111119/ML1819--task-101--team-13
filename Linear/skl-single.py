#!/usr/bin/env python
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import numpy as np


def normaliseData(x):
  # rescale data to lie between 0 and 1
  scale = x.max()
  return (x/scale)


#Load data
iris = datasets.load_iris()

# choose petal length and petal width as feature via chi-squared test.
X = SelectKBest(chi2, k=1).fit_transform(iris.data, iris.target)
y = np.array([x[3] for x in iris.data])

#Split the train and test data
train_X, test_X, train_Y, test_Y = train_test_split(X, y,test_size=0.3, random_state=None)

#Normalize
X_train = normaliseData(train_X)
X_test = normaliseData(test_X)
y_train = np.reshape(train_Y, [len(train_Y), 1])
y_test = np.reshape(test_Y, [len(test_Y), 1])


#fit the model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)
print("Mean squared error: %.5f" % mean_squared_error(y_test, y_pred))
#print('Variance score: %.5f' % r2_score(price_y_train, regr.coef_*price_X_train + regr.intercept_))
print('Variance score: %.5f' % r2_score(y_test, y_pred))
#print(price_y_test)
#print(price_y_pred)

#plot
plt.plot(X_train, y_train, 'ro', label='Data Points')
plt.plot(X_train, regr.coef_*X_train + regr.intercept_, label='Best fit line')
plt.legend(loc='upper left')
plt.title('Petal Length and Petal Width (scikit-learn)')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()