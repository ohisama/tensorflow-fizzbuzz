import numpy as np
import tensorflow as tf

def binary_encode(i, j):
	return np.array([i >> d & 1 for d in range(j)])
def fizz_buzz_encode(i):
	if i % 15 == 0:
		return np.array([0, 0, 0, 1])
	elif i % 5 == 0:
		return np.array([0, 0, 1, 0])
	elif i % 3 == 0:
		return np.array([0, 1, 0, 0])
	else:
		return np.array([1, 0, 0, 0])
trX = np.array([binary_encode(i, 10) for i in range(101, 2 ** 10)])
trY = np.array([fizz_buzz_encode(i) for i in range(101, 2 ** 10)])
print(trX)
print(trY)

X = tf.placeholder("float", [None, 10])
Y = tf.placeholder("float", [None, 4])

w_h = tf.Variable(tf.random_normal([10, 100], stddev = 0.01))
h = tf.nn.relu(tf.matmul(X, w_h))
w_o = tf.Variable(tf.random_normal([100, 4], stddev = 0.01))
out = tf.matmul(h, w_o)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out, labels = Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
predict_op = tf.argmax(out, 1)

def fizz_buzz(i, prediction):
	return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

with tf.Session() as sess:
	tf.initialize_all_variables().run()
	for epoch in range(10000):
		p = np.random.permutation(range(len(trX)))
		trX, trY = trX[p], trY[p]
		for start in range(0, len(trX), 128):
			end = start + 128
			sess.run(train_op, feed_dict = {
				X: trX[start:end],
				Y: trY[start:end]
			})
		if epoch % 500 == 0:
			print(epoch, np.mean(np.argmax(trY, axis = 1) == sess.run(predict_op, feed_dict = {
				X: trX,
				Y: trY
			})))
	numbers = np.arange(1, 101)
	teX = np.transpose(binary_encode(numbers, 10))
	teY = sess.run(predict_op, feed_dict = {
		X: teX
	})
	output = np.vectorize(fizz_buzz)(numbers, teY)
	print(output)



