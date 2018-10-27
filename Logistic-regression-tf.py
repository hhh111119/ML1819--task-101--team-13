import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from util.data import load_data
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


class Classifer:
    pass


def main():
    X_train, X_test, y_plt_train, y_plt_test, y_train, y_test = load_data(
        'iris.data')

    X = tf.placeholder(tf.float32, [None, 2])
    y = tf.placeholder(tf.float32, [None, 3])

    w = tf.Variable(tf.random_normal(
        [2, 3], mean=0.0, stddev=1.0), trainable=True, dtype=tf.float32)
    b = tf.Variable(tf.zeros([3]), trainable=True)
    y_pred = tf.add(tf.matmul(X, w), b)
    y_prab = tf.nn.softmax(tf.add(tf.matmul(X, w), b))

    res = tf.argmax(y_prab, 1)
    classfier = Classifer()
    classfier.predict = res
    cost = -tf.reduce_sum(y * tf.log(y_prab))
    learning_rate = 0.01
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y_prab, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()

    with tf.Session() as session:
        print('start')
        session.run(init)
        X_tr, y_tr = X_train, y_train
        bach = {X: X_tr, y: y_tr}
        bach_size = 1000
        for i in range(bach_size):

            train.run(bach)
            if i % (bach_size/10) == 0:
                cur_accuracy = accuracy.eval(bach)
                print('times %5d : accuracy = %8.3f' % (i, cur_accuracy))

        bach = {X: X_test, y: y_test}
        final_accuracy = accuracy.eval(bach)
        print('accuracy = %8.3f' % final_accuracy)
        print(res.eval(bach))
        ws = session.run(w)
        bs = session.run(b)
        print(ws)
        print(bs)
        X_a = np.vstack((X_train, X_test))
        y_a = np.hstack((y_plt_train, y_plt_test))


        resolution=0.02
        test_idx=range(105, 150)
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y_a))])
        # plot the decision surface
        x1_min, x1_max = X_a[:, 0].min() - 1, X_a[:, 0].max() + 1
        x2_min, x2_max = X_a[:, 1].min() - 1, X_a[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
        Z = res.eval({X:np.array([xx1.ravel(), xx2.ravel()],dtype="float32").T})
        #bach = {X:X, y:y}
        #print('bach x', X)
        #Z = classifier.eval()
        Z = Z.reshape(xx1.shape)
        print(Z)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        # plot class samples
        for idx, cl in enumerate(np.unique(y_a)):
            plt.scatter(x=X_a[y_a == cl, 0], y=X_a[y_a == cl, 1], alpha=0.8, c=cmap(idx),
                        marker=markers[idx], label=cl)
        # highlight test samples
        if test_idx:
            X_test, y_test = X_a[test_idx, :], y_a[test_idx]
            plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1,
                        marker='o', s=55, label='test set')

        #plot_decision_regions(X_combined_std, y_combined, classifier=res,test_idx=range(105, 150))
        plt.show()


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    #Z = classifier.eval({X:np.array([xx1.ravel(), xx2.ravel()],dtype="float32").T})
    #bach = {X:X, y:y}
    print('bach x', X)
    Z = classifier.eval()
    Z = Z.reshape(xx1.shape)
    print(Z)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1,
                    marker='o', s=55, label='test set')


main()
