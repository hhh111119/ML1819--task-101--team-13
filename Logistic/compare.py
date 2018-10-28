from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from util.data import load_data
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt





def sk_l(X_train, X_test, y_train, y_test, y_tf_train, y_tf_test):
    
    # 为了追求机器学习和最优化算法的最佳性能，进行特征缩放
  

    lr = LogisticRegression(C=1000.0, random_state=0)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print('accuracy:' + str(score))

    
    return score

def tf_l( X_train, X_test, y_plt_train, y_plt_test, y_train, y_test):
   
    
    X = tf.placeholder(tf.float32, [None, 2])
    y = tf.placeholder(tf.float32, [None, 3])

    w = tf.Variable(tf.random_normal(
        [2, 3], mean=0.0, stddev=1.0), trainable=True, dtype=tf.float32)
    b = tf.Variable(tf.zeros([3]), trainable=True)
    y_pred = tf.add(tf.matmul(X, w), b)
    y_prab = tf.nn.softmax(tf.add(tf.matmul(X, w), b))

    res = tf.argmax(y_prab, 1)
    
    cost = -tf.reduce_sum(y * tf.log(y_prab))
    learning_rate = 0.01
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y_prab, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()

    cost_values = []
    with tf.Session() as session:
        print('start')
        session.run(init)
        X_tr, y_tr = X_train, y_train
        bach = {X: X_tr, y: y_tr}
        bach_size = 3000
        for i in range(bach_size):

            train.run(bach)
            cost_values.append(cost.eval(bach))
            if i % (bach_size/10) == 0:
                cur_accuracy = accuracy.eval(bach)

                #print('times %5d : accuracy = %8.3f' % (i, cur_accuracy))

        bach = {X: X_test, y: y_test}
        final_accuracy = accuracy.eval(bach)
        print('accuracy = %f' % final_accuracy)
        #print(res.eval(bach))
        return final_accuracy

def main():
    tf_accuracies = []
    sk_accuracies = []
    times = []
    counts = 500
    for i in range(counts):
         X_train, X_test, y_plt_train, y_plt_test, y_train, y_test = load_data('iris.data')
         tf_ac = tf_l( X_train, X_test, y_plt_train, y_plt_test, y_train, y_test)
         sk_ac = sk_l( X_train, X_test, y_plt_train, y_plt_test, y_train, y_test)
         tf_accuracies.append(tf_ac)
         sk_accuracies.append(sk_ac)
         times.append(i)
    
    #print(tf_accuracies)
    #print(sk_accuracies)
    tf_large = 0
    equal = 0
    tf_small = 0
    for i in range(counts):
        if((tf_accuracies[i] - sk_accuracies[i])>1e-5):
            tf_large += 1
        elif(abs((tf_accuracies[i] - sk_accuracies[i])) <= 1e-5):
            equal += 1
        else:
            tf_small += 1
    print()
    print('tesorflow better:'+str(tf_large)+' equal:'+ str(equal)+' sk better:'+str(tf_small))
    plt.gca().set_color_cycle(['red', 'green'])
    plt.plot(times, tf_accuracies)
    plt.plot(times, sk_accuracies)
    plt.xticks(np.arange(min(times), max(times)+1, 1.0))
    plt.xlabel('test id')
    plt.ylabel('accuracy')
    plt.legend(['tensorflow accuracy', 'scikit-learn accuracy'])
    plt.grid(True)
    plt.savefig('comare'+str(counts)+'.png')
    plt.show()


main()
