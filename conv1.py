from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 
import tensorflow.compat.v1 as tfc

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tfc.InteractiveSession()

def weight_variable(shape):
    initial = tfc.truncated_normal(shape, stddev=0.1)
    return tf.compat.v1.Variable(initial)

def bias_variable(shape):
#    initial = tfc.constant(0.1, shape=shape)
    initial = tf.compat.v1.constant(0.1, shape=shape)
#    return tfc.Variable(initial)
    return tf.compat.v1.Variable(initial)

def conv2d(x, W):
    return tf.compat.v1.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.compat.v1.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    
    x = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 784])
    y_ = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 10])
    
    x_image = tf.compat.v1.reshape(x, [-1,28,28,1])
    
    # 1st convolution layer
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.compat.v1.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    # 2nd convolution layer
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.compat.v1.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    # Fully Connected
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.compat.v1.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.compat.v1.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    #Dropout
    keep_prob = tf.compat.v1.placeholder(tf.float32)
    h_fc1_drop = tf.compat.v1.nn.dropout(h_fc1, keep_prob)
    
    #Final layer - Softmax
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.compat.v1.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    #Cross_Entropy
    cross_entropy = tf.compat.v1.reduce_mean(-tf.compat.v1.reduce_sum(y_ * tf.compat.v1.log(y_conv), reduction_indices=[1]))
    train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    #Prediction Accuracy
    correct_prediction = tf.compat.v1.equal(tf.compat.v1.arg_max(y_conv, 1), tf.compat.v1.arg_max(y_, 1))
    accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_prediction, tf.float32))
    
    # Train
    tf.compat.v1.global_variables_initializer().run()
    
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        
        if i % 100 == 0:            
            train_accuracy = accuracy.eval(feed_dict = {x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            
    # Final Test
    print( "Test accuracy %g" % accuracy.eval(feed_dict = {x: mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0}))
    
