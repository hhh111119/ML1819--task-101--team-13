import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler

def load_data_multi(filename):
    with open(filename) as i:
        lines = i.readlines()
    X = []
    Y = []
   # Y_tf = []
  #  replace_y = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
   # replace_y_tf = {'Iris-setosa':[1,0,0], 'Iris-versicolor':[0,1,0], 'Iris-virginica':[0,0,1]}
    for line in lines:
        x1, x2, x3,x4, y1 = line.split(',')
        x = []
        x.append(x1)
        x.append(x2)
        x.append(x3)
        X.append(x)
        Y.append(x4)
       # Y_tf.append(replace_y_tf[y1.strip()])

    #X = SelectKBest(chi2, k=2).fit_transform(X, Y)
    np_X = np.array(X,dtype=np.float32)
    np_Y = np.array(Y,dtype=np.float32)
  #  np_tf_Y = np.array(Y_tf)
   # np_X = SelectKBest(chi2, k=3).fit_transform(np_X, np_Y)
    

    return train_test_split(np_X, np_Y)


def train_test_split(X, y, test_ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ration must be valid"

    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]
  #  y_tf_train = y_tf[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]
 #   y_tf_test = y_tf[test_indexes]
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test
    


