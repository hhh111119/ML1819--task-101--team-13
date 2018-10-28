#!/usr/bin/env python
from sklearn import linear_model
from sklearn.datasets import load_boston
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

price=load_boston()
#print price.data.shape  (m,n)=(506,13)
#print price.feature_names ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
#print price.target
def normaliseData(x):
  # rescale data to lie between 0 and 1
  scale = x.max()
  return (x/scale)

#price_X_train, price_X_test, price_y_train, price_y_test = train_test_split(price.data, price.target, random_state=1)
print("price_x")
#print(price_X_train)
# price_X_train = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
# price_X_test = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
# price_y_train = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
# price_y_test = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
# price_X_test = price_X_test.reshape(-1, 1)
# price_X_train = price_X_train.reshape(-1, 1)
iris = datasets.load_iris()
price_X_train = np.array([x[2] for x in iris.data])

x1 = np.array(price_X_train[0:30])
xt1 = price_X_train[30:40]
x2 = price_X_train[40:70]
xt2 = price_X_train[70:80]
x3 = price_X_train[80:110]
xt3 = price_X_train[110:120]
x4 = price_X_train[120:150]
price_X_train = np.append(np.append(np.append(x1, x2), x3), x4)
price_X_test = np.append(np.append(xt1,xt2), xt3)


X_test = np.array([x[3] for x in iris.data])
y1 = np.array(X_test[0:30])
yt1 = X_test[30:40]
y2 = X_test[40:70]
yt2 = X_test[70:80]
y3 = X_test[80:110]
yt3 = X_test[110:120]
y4 = X_test[120:150]
price_y_train = np.append(np.append(np.append(y1, y2), y3), y4)
price_y_test = np.append(np.append(yt1,yt2), yt3)
price_X_train = normaliseData(price_X_train).reshape(-1, 1)
price_y_train = normaliseData(price_y_train).reshape(-1, 1)


'''
price_X_test = price_X_train[0:20]
price_X_train = price_X_train[20:150]
price_X_train = np.reshape(price_X_train, [len(price_X_train), 1])
price_y_train = np.array([y[3] for y in iris.data])

price_y_test = price_y_train[0:20]
price_y_train = price_y_train[20:150]
price_y_train = np.reshape(price_y_train, [len(price_y_train), 1])
'''
print(price_X_train.shape)
print(price_y_train.shape)
print("price_X_test:", price_X_test.shape)
print("price_y_test:", price_y_test.shape)


price_X_test = normaliseData(price_X_test).reshape(-1, 1)
price_y_test = normaliseData(price_y_test).reshape(-1, 1)





print(price_X_train.shape) #(379,13)
print(price_y_train.shape)
print(price_X_test.shape) #(127,13)
print(price_y_test.shape)
regr = linear_model.LinearRegression()
regr.fit(price_X_train, price_y_train)

price_y_pred = regr.predict(price_X_test)

print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)
print("Mean squared error: %.5f" % mean_squared_error(price_y_test, price_y_pred))
#print('Variance score: %.5f' % r2_score(price_y_train, regr.coef_*price_X_train + regr.intercept_))
print('Variance score: %.5f' % r2_score(price_y_test, price_y_pred))
#print(price_y_test)
#print(price_y_pred)

#plot
plt.plot(price_X_train, price_y_train, 'ro', label='Data Points')
plt.plot(price_X_train, regr.coef_*price_X_train + regr.intercept_, label='Best fit line')
plt.legend(loc='upper left')
plt.title('Petal Length and Petal Width (scikit-learn)')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()