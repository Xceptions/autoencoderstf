import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

"""
    building a simple Principal Component Assembly using autoencoding
"""

df = pd.read_csv("/home/exceptions/datasets/Incompressible_linear_data.csv", header=0)

drop_cols = ['Flow Regime', 'Fluid Type', 'Reservoir geometry']
train = df.drop(drop_cols, axis=1)
X_train, X_test = train_test_split(train, random_state=0)


n_input = 8
n_hidden = 5
n_output = n_input
learning_rate = 0.01


X = tf.placeholder(tf.float32, shape=[None, n_input])

with tf.name_scope("dnn"):
    hidden = tf.layers.dense(X, n_hidden, name="hidden")
    outputs = tf.layers.dense(hidden, n_output, name="output")


with tf.name_scope("loss"):
    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) # we are comparing the outputs against X using mse


with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()

n_iterations = 1000
codings = hidden

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        sess.run(training_op, feed_dict={X: X_train}) # no labels (unsupervised)
    codings_val = codings.eval(feed_dict={X: X_test})

    print(codings_val)