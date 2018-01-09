# -*- coding:utf-8 -*-

import tensorflow as tf
import fetalheart_func

tf.set_random_seed(0)

# 读入数据，并划分为fatalheart.train(80%的数据) fatalheart.test(20%)
fetalheart = read_data_sets('../data/', one+hot=True, reshape=False, validation_size=0)

sess = tf.InteractiveSession()
# input X: 151*2402 images
X = tf.placeholder(tf.int8, [None, 151, 2404, 1])
# weights W[151*2404, 3]
W = tf.Variable(tf.zeros([151*2404, 3]))
# biases b[3]
b = tf.Variable(tf.zeros([3]))
# flat the image matrix to 1D
XX = tf.reshape(X, [-1, 151*2402])

# labels
Y_ = tf.placeholder(tf.int8, [None, 3])
# The model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y), reduction_indices=[1]))

# set learning_rate=0.05
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# 全局参数初始化
tf.global_variables_initializer().run()

# 迭代执行训练操作，每次随机抽取100条样本构成mini-batch
for i in range(100):
    batch_xs, batch_ys = fetalheart.train.next_batch(100)
    train_step.run({X: batch_xs, Y_:batch_ys})

# accuracy of the model
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("max test accuracy:%s"%accuracy)