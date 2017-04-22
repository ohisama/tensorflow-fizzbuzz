import tensorflow as tf
import numpy as np

def rand01(digit):
	s = abs(np.random.normal(0.0, 0.05))
	if digit == 0:
		noise = digit + s
	else:
		noise = digit - s
	return noise
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
add_noise = np.vectorize(rand01)
X = add_noise(X)
Y = add_noise(Y)
print (X)
print (Y)

x_ = tf.placeholder(tf.float32, shape = [4, 2])
y_ = tf.placeholder(tf.float32, shape = [4, 1])

w1 = tf.Variable(tf.random_uniform([2, 5], -1, 1))
w2 = tf.Variable(tf.random_uniform([5, 1], -1, 1))
b1 = tf.Variable(tf.zeros([5]))
b2 = tf.Variable(tf.zeros([1]))
layer1 = tf.tanh(tf.matmul(x_, w1) + b1)
out = tf.tanh(tf.matmul(layer1, w2) + b2)
out = tf.add(out, 1)
out = tf.multiply(out, 0.5)
loss = tf.reduce_mean(tf.square(Y - out))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(4001):
	sess.run(train_step, feed_dict = {
		x_: X,
		y_: Y
	})
	if i % 2000 == 0:
		print('Batch ', i)
		print('loss ', sess.run(loss, feed_dict = {
			x_: X,
			y_: Y
		}))
		print('Inference ', sess.run(out, feed_dict = {
			x_: X,
			y_: Y
		}))


