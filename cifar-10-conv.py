import numpy as np
import cifar10, cifar10_input
import tensorflow as tf
import time

max_steps = 3000
batch_size = 128
data_dir = '/media/data/tf_data/cifar-10-batches-py'

def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.random.truncated_normal(shape, stddev = stddev))
    
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name = 'weight_loss')
        tf.compat.v1.add_to_collection('losses', weight_loss)
        
    return var

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool2d (x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

images_train, labels_train = cifar10_input.distorted_inputs(batch_size=batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data=True, batch_size=batch_size)

# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.get_default_graph()

image_holder = tf.compat.v1.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.compat.v1.placeholder(tf.int32, [batch_size])

#1st Conv Layer
weight1 = variable_with_weight_loss(shape=[5,5,3,64], stddev=5e-2, wl=0.0)
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(image_holder, weight1), bias1))
pool1 = max_pool_3x3(conv1)
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75)

#2nd Conv Layer
weight2 = variable_with_weight_loss(shape=[5,5,64,64], stddev=5e-2, wl=0.0)
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(norm1, weight2), bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75)
pool2 = max_pool_3x3(norm2)

#1st Fully Connected
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1]
weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl = 0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshape, weight3),bias3))

#2nd Fully Connected
weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.nn.bias_add(tf.matmul(local3, weight4), bias4))

#Final layer
weight5 = variable_with_weight_loss(shape=[192,10], stddev=1/192.0, wl =0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)

# Calculate Loss
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = logits, labels = labels, name = 'cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')

    tf.compat.v1.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.compat.v1.get_collection('losses'), name = 'total_loss')

loss = loss(logits, label_holder)
train_op = tf.compat.v1.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

# Session
sess = tf.compat.v1.InteractiveSession()
tf.compat.v1.global_variables_initializer().run()

# Train
tf.train.start_queue_runners()

for step in range(max_steps):
    start_time = time.time()

    image_batch, label_bach = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss],
                             feed_dict = {image_holder: image_batch, label_holder: label_bach}
                    )
    duration = time.time() - start_time
    
    if step % 10 == 0:
        example_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        format_str = ('step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step, loss_value, example_per_sec, sec_per_batch))

# Test
num_examples = 10000
import math
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size

for step in range(num_iter):
    image_batch, label_bach = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op],
                           feed_dict={image_holder: image_batch, label_holder:label_bach})
    true_count += np.sum(predictions)
    step += 1
    
# Accuracy
print('total count: %d, true count: %d' % (total_sample_count, true_count))
precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)