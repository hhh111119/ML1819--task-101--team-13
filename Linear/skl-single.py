#!/usr/bin/env python
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import numpy as np
import time 
from util.data import load_data


#calculate the time
start_time = time.time() 


# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
# Load data and normalize
train_X, test_X, train_Y, test_Y = load_data('iris.data')
#train_Y = np.reshape(train_Y, [len(train_Y), 1])
#test_Y = np.reshape(test_Y, [len(test_Y), 1])



#fit the model
regr = linear_model.LinearRegression()
regr.fit(train_X, train_Y)

y_pred = regr.predict(test_X)

print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)
print("Mean squared error: %.5f" % mean_squared_error(test_Y, y_pred))
#print('Variance score: %.5f' % r2_score(price_y_train, regr.coef_*price_X_train + regr.intercept_))
print('R²_score: %.5f' % r2_score(test_Y, y_pred))
#print(price_y_test)
#print(price_y_pred)

#plot
plt.plot(test_X, test_Y, 'ro', label='Data Points')
plt.plot(test_X, regr.coef_*test_X + regr.intercept_, label='Best fit line')
plt.legend(loc='upper left')
plt.title('Petal Length and Petal Width (scikit-learn)')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))