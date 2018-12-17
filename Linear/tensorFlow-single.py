#----------------------------------
# This function shows how to use TensorFlow to
# solve linear regression.
# y = Ax + b

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from util.data import load_data
from tensorflow.python.framework import ops
from sklearn.metrics import mean_squared_error, r2_score
import time 

rng = np.random

#calculate the time
start_time = time.time() 


ops.reset_default_graph()

# Create graph
sess = tf.Session()


# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
# Load data and normalize
train_X, test_X, train_Y, test_Y = load_data('iris.data')
train_Y = np.reshape(train_Y, [len(train_Y), 1])
test_Y = np.reshape(test_Y, [len(test_Y), 1])


#x_vals = np.array([x[2] for x in iris.data])
#x_test = np.append(np.append(x_vals[30:40], x_vals[70:80]), x_vals[110:120])
#x_vals = np.append(np.append(np.append(x_vals[0:30], x_vals[40:70]), x_vals[80:110]), x_vals[120: 150])
#x_vals = normaliseData(x_vals)

#y_vals = np.array([y[3] for y in iris.data])
#yt_est = np.append(np.append(y_vals[30:40], y_vals[70:80]), y_vals[110:120])
#y_vals = np.append(np.append(np.append(y_vals[0:30], y_vals[40:70]), y_vals[80:110]), y_vals[120: 150])
#y_vals = normaliseData(y_vals)

#print(x_vals.shape)
#print(y_vals.shape)

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
    if (i+1)%30==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))

# Get A and b
[slope] = sess.run(A)
[y_intercept] = sess.run(b)

# best_fit line
best_fit = []
for i in test_X:
  best_fit.append(slope*i+y_intercept)

#fit line
plt.plot(test_X, test_Y, 'o', label='Data Points')
plt.plot(test_X, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Petal Length and Petal Width (TensorFlow)')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('L2 Loss per Generation (TensorFlow)')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.show()

# Prediction Error
error = test_Y - best_fit
plt.hist(error, bins=10)
plt.xlabel("Prediction Error [1000$]")
plt.show()
_ = plt.ylabel("Count")

print("Mean squared error: %.5f" % mean_squared_error(test_Y, best_fit))
print('R²_score: %.5f' % r2_score(test_Y, best_fit))

print("--- %s seconds ---" % (time.time() - start_time))