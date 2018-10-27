import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from util.data import load_data

def main():
    X_tran, X_test, _,_,y_train, y_test = load_data('iris.data')

    X = tf.placeholder(tf.float32, [None, 2])
    y = tf.placeholder(tf.float32, [None, 3])

    w = tf.Variable(tf.random_normal([2, 3], mean=0.0, stddev=1.0),trainable=True,dtype=tf.float32)
    b = tf.Variable(tf.zeros([3]),trainable=True)

    y_prab = tf.nn.softmax(tf.add(tf.matmul(X, w), b))
    cost = -tf.reduce_sum(y * tf.log(y_prab))
    learning_rate = 0.01
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y_prab, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    init = tf.initialize_all_variables()

    with tf.Session() as session:
        print('start')
        session.run(init)
        X_tr, y_tr = X_tran, y_train
        bach = {X: X_tr, y: y_tr }
        bach_size = 1000
        for i in range(bach_size):
            
            
            train.run(bach)
            if i % (bach_size/10) == 0:
                cur_accuracy = accuracy.eval(bach)
                print('times %5d : accuracy = %8.3f' %(i, cur_accuracy))
        
        bach = {X: X_test, y: y_test }
        final_accuracy = accuracy.eval(bach)
        print('accuracy = %8.3f' % final_accuracy)      




main()