import tensorflow as tf
import numpy as np
import math, random
import matplotlib.pyplot as plt

np.random.seed(1000)
function_to_learn = lambda x: np.sin(x) + 0.1 * np.random.randn(*x.shape)
all_x = np.float32(np.random.uniform(-2 * math.pi, 2 * math.pi, (1, 1000))).T
np.random.shuffle(all_x)
train_size = int(1000 * 0.8)
trainx = all_x[:train_size]
validx = all_x[train_size:]
trainy = function_to_learn(trainx)
validy = function_to_learn(validx)

X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32, [None, 1])

low0 = -4 * np.sqrt(6.0 / (1 + 10))
high0 = 4 * np.sqrt(6.0 / (1 + 10))
w_h = tf.Variable(tf.random_uniform([1, 10], minval = low0, maxval = high0, dtype = tf.float32))
b_h = tf.Variable(tf.zeros([1, 10], dtype = tf.float32))
h = tf.nn.sigmoid(tf.matmul(X, w_h) + b_h)
low1 = -4 * np.sqrt(6.0 / (10 + 1))
high1 = 4 * np.sqrt(6.0 / (10 + 1))
w_o = tf.Variable(tf.random_uniform([10, 1], minval = low1, maxval = high1, dtype = tf.float32))
b_o = tf.Variable(tf.zeros([1, 1], dtype = tf.float32))
out = tf.matmul(h, w_o) + b_o
loss = tf.nn.l2_loss(out - Y)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
errors = []
for i in range(1000):
	for start, end in zip(range(0, len(trainx), 100), range(100, len(trainx), 100)):
		sess.run(train_op, feed_dict = {
			X: trainx[start:end],
			Y: trainy[start:end]
		})
	mse = sess.run(tf.nn.l2_loss(out - validy), feed_dict = {
		X: validx
	})
	errors.append(mse)
	if i % 500 == 0:
		print ("epoch %d, validation MSE %g" % (i, mse))
yy = sess.run(out, feed_dict = {
	X: validx
})
plt.scatter(validx, yy)
plt.show()


