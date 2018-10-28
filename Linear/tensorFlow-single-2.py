#----------------------------------
# This function shows how to use TensorFlow to
# solve linear regression.
# y = Ax + b

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
from sklearn.metrics import mean_squared_error, r2_score

ops.reset_default_graph()

def normaliseData(x):
  # rescale data to lie between 0 and 1
  scale = x.max()
  return (x/scale)

# Create graph
sess = tf.Session()

# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([x[2] for x in iris.data])
x_test = np.append(np.append(x_vals[30:40], x_vals[70:80]), x_vals[110:120])
x_vals = np.append(np.append(np.append(x_vals[0:30], x_vals[40:70]), x_vals[80:110]), x_vals[120: 150])
x_vals = normaliseData(x_vals)

y_vals = np.array([y[3] for y in iris.data])
yt_est = np.append(np.append(y_vals[30:40], y_vals[70:80]), y_vals[110:120])
y_vals = np.append(np.append(np.append(y_vals[0:30], y_vals[40:70]), y_vals[80:110]), y_vals[120: 150])
y_vals = normaliseData(y_vals)

print(x_vals.shape)
print(y_vals.shape)

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
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    if (i+1)%30==0:
        print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))

# Get A and b
[slope] = sess.run(A)
[y_intercept] = sess.run(b)

# best_fit line
best_fit = []
for i in x_vals:
  best_fit.append(slope*i+y_intercept)

#fit line
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
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
error = y_vals - best_fit
plt.hist(error, bins=10)
plt.xlabel("Prediction Error [1000$]")
plt.show()
_ = plt.ylabel("Count")

print("Mean squared error: %.5f" % mean_squared_error(y_vals, best_fit))
print('Variance score: %.5f' % r2_score(y_vals, best_fit))