from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
from sklearn.metrics import r2_score
from sklearn import datasets

rng = np.random

# Parameters
learning_rate = 0.01
training_epochs = 2000
display_step = 50

# Rescale data to lie between 0 and 1
def normaliseData(x): 
  scale = x.max()
  return (x/scale)

# Load data and normalize
iris = datasets.load_iris()
x_vals = np.array([x[0:3] for x in iris.data])
test_X = np.append(np.append(x_vals[30:40], x_vals[70:80],axis=0), x_vals[110:120],axis=0)
test_X = normaliseData(test_X)
train_X = np.append(np.append(np.append(x_vals[0:30], x_vals[40:70],axis=0), x_vals[80:110],axis=0), x_vals[120: 150],axis=0)
train_X = normaliseData(train_X)

y_vals = np.array([y[3] for y in iris.data])
test_Y = np.append(np.append(y_vals[30:40], y_vals[70:80]), y_vals[110:120])
test_Y = normaliseData(test_Y)
train_Y = np.append(np.append(np.append(y_vals[0:30], y_vals[40:70]), y_vals[80:110]), y_vals[120: 150])
train_Y = normaliseData(train_Y)
train_Y = np.reshape(train_Y, [len(train_Y), 1])
test_Y = np.reshape(test_Y, [len(test_Y), 1])

print(train_X.shape)
print(train_Y.shape)


# tf Graph Input
X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 1])

# Set model weights
W = tf.Variable(tf.zeros([3, 1]), dtype=tf.float32)
b = tf.Variable(np.zeros([1]), dtype=tf.float32)
W_final=np.zeros(3)
b_final=0

# Construct a linear model
#pred = tf.add(tf.multiply(X, W), b)
pred = tf.matmul(X,W) + b

# Mean squared error
cost = tf.reduce_mean(tf.square(Y-pred))

# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
#init = tf.global_variables_initializer()
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})
for j in range(3):
   W_final[j]=sess.run(W[j])
b_final=sess.run(b)
# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})

    # Testing example, as requested (Issue #2)

    #print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),feed_dict={X: test_X,Y: test_Y})  # same function as cost above
    #print("Testing cost=", testing_cost)
    #print("Absolute mean square loss difference:", abs(training_cost - testing_cost))

    #print(("ttt: %.5f" % tf.matmul(test_X, sess.run(W))+ sess.run(b)))
    print("mean_squared_error:%.5f" % np.mean((np.reshape(test_Y, [1,len(test_Y)]) - (np.dot(test_X,W_final)+b_final)) ** 2))
    print("Variance score: %.5f" % r2_score(test_Y, np.dot(test_X,W_final)+b_final))