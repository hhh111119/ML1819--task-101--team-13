import numpy as np
def load_data(filename):
    with open(filename) as i:
        lines = i.readlines()
    X = []
    Y = []
    for line in lines:
        x1, x2, x3,x4, y1 = line.split(',')
        x = []
        x.append(x1)
        x.append(x2)
        x.append(x3)
        x.append(x4)
        X.append(x)
        Y.append(y1)

    np_X = np.array(X)
    np_Y = np.array(Y)
    #print(np_X)
    #print(np_Y)
    #print(np_arr)
    return train_test_split(np_X, np_Y)


def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据 X 和 y 按照test_ratio分割成X_train, X_test, y_train, y_test"""
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

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test
    

