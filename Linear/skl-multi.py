#!/usr/bin/env python
from sklearn import linear_model
from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import numpy as np
sc = StandardScaler() 


# Rescale data to lie between 0 and 1
def normaliseData(x): 
  scale = x.max()
  return (x/scale)

#Load data
data=boston_housing.load_data()
X=np.append(data[0][0],data[1][0],axis=0)
Y=np.append(data[0][1],data[1][1],axis=0)
Y=np.array(Y, dtype=int)

#Choose the best k features
X = SelectKBest(chi2, k=13).fit_transform(X, Y)

#Split the train and test data
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, random_state=1)

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
print('Variance score: %.2f' % r2_score(test_Y, pred))

plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13], [0.19,0.09,0.1,0.1,0.11,0.09,0.49,0.48,0.55,0.56,0.61,0.70,0.71], 'ro','b-', label='Best fit line', linewidth=3)
plt.title('R²-n features')
plt.xlabel('Amount of feature')
plt.ylabel('R²')
plt.show()

   