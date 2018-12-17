from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from tensorflow.python.framework import ops
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from util.data import load_data
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import time 






def sk_linear(train_X, test_X, train_Y, test_Y):
    
    start_time = time.time() 
    #fit the model
    regr = linear_model.LinearRegression()
    regr.fit(train_X, train_Y)

    y_pred = regr.predict(test_X)
    runtime = (time.time() - start_time)

    return mean_squared_error(test_Y, y_pred), runtime



def tf_linear(train_X, test_X,train_Y, test_Y):
    start_time = time.time() 
    sess = tf.Session()   
    train_Y = np.reshape(train_Y, [len(train_Y), 1])
    test_Y = np.reshape(test_Y, [len(test_Y), 1])
    batch_size = 25

# Initialize
    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# variavles
    A = tf.Variable(tf.random_normal(shape=[1,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))

# model，y=Ax+b
    model_output = tf.add(tf.matmul(x_data, A), b)

# Loss of L2
    loss = tf.reduce_mean(tf.square(y_target - model_output))

# learning rate
    my_opt = tf.train.GradientDescentOptimizer(0.05)
    train_step = my_opt.minimize(loss)

# init
    init = tf.global_variables_initializer()
    sess.run(init)

# iteration
    loss_vec = []
    for i in range(1000):
        rand_index = np.random.choice(len(train_X), size=batch_size)
        sess.run(train_step, feed_dict={x_data: train_X, y_target: train_Y})
        temp_loss = sess.run(loss, feed_dict={x_data: train_X, y_target: train_Y})
        loss_vec.append(temp_loss)

# Get A and b
    [slope] = sess.run(A)
    [y_intercept] = sess.run(b)

# best_fit line
    best_fit = []
    for i in test_X:
      best_fit.append(slope*i+y_intercept)
    runtime = (time.time() - start_time)

    return mean_squared_error(test_Y, best_fit), runtime

def main():
    tf_MSE = []
    sk_MSE = []
    tf_RUNTIME = []
    sk_RUNTIME = []
    times = []
    counts = 25
    for i in range(counts):
         X_train, X_test,y_train, y_test = load_data('iris.data')
         tf_mse,tf_runtime = tf_linear( X_train, X_test, y_train, y_test)
         sk_mse,sk_runtime = sk_linear( X_train, X_test, y_train, y_test)
         tf_MSE.append(tf_mse)
         sk_MSE.append(sk_mse)
         tf_RUNTIME.append(tf_runtime)
         sk_RUNTIME.append(sk_runtime)
         times.append(i)
    

    #print('tesorflow better:'+str(tf_large)+' equal:'+ str(equal)+' sk better:'+str(tf_small))
 #   plt.gca().set_color_cycle(['red', 'green'])
    plt.plot(times, tf_MSE)
    plt.plot(times, sk_MSE)
    plt.xticks(np.arange(min(times), max(times)+1, 1.0))
    plt.xlabel('test id')
    plt.ylabel('mean square error')
    plt.legend(['tensorflow mean square error', 'scikit-learn mean square error'])
    plt.grid(True)
    plt.savefig('comare'+str(counts)+'.png')
    plt.show()
    
    plt.plot(times, tf_RUNTIME)
    plt.plot(times, sk_RUNTIME)
    plt.xticks(np.arange(min(times), max(times)+1, 1.0))
    plt.xlabel('test id')
    plt.ylabel('run time')
    plt.legend(['tensorflow run time', 'scikit-learn run time'])
    plt.grid(True)
    plt.savefig('comare'+str(counts)+'.png')
    plt.show()

main()
