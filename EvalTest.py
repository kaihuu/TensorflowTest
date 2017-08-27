import tensorflow as tf
from sklearn import datasets

N = 300
X, y = datasets.make_moons(N, noise=0.3)

from sklearn.model_selection import train_test_split

Y = y.reshape(N, 1)
X_train, X_test, Y_train, Y_test =\
    train_test_split(X, Y, train_size=0.8)

num_hidden = 2

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])