#Constant values are stored in the graph definition
#Sessions allocate memory to store variable values
#Feed values into placeholders with a dictionary (feed_dict)
#Easy to use but poor performance
#Separate the assembling of graph and executing ops
#Use Python attribute to ensure a function is only loaded the first time it’s called

""" Starter code for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
import utils

# Define paramaters for the model
learning_rate = 0.001
batch_size = 1024
n_epochs = 20
n_train = 60000
n_test = 10000

#TODO: EXPLAIN THIS
# Step 1: Read in data
mnist_folder = 'datasets/MNIST_data'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)# Reads in each dataset. val contains train_data labels

# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000) # if you want to shuffle your data
train_data = train_data.batch(batch_size)#Groups data to spped up Training

# create testing Dataset and batch it
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.shuffle(10000) # if you want to shuffle your data
test_data = test_data.batch(batch_size)


# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                           train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)	# initializer for train_data
test_init = iterator.make_initializer(test_data)	# initializer for train_data

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y

# X = tf.get_variable(name='feature', initializer=tf.zeros([784]))
# w=tf.get_variable(name='weight',initializer=tf.random_normal([784,10],mean=0,stddev=0.01))
# b=tf.get_variable(name='bias',initializer=tf.zeros([10]))
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(img, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, 1.0)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])


# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# logits = (tf.matmul(img,w) + b)
logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2



# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label,logits=logits)


# Step 6: define optimizer
# using Adamn Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./logdir/logreg', tf.get_default_graph())
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 14
config.inter_op_parallelism_threads = 14
# tf.summary.scalar("loss", self.loss)
# tf.summary.scalar("accuracy", self.accuracy)
# tf.summary.histogram("histogram loss", self.loss)
# # because you have several summaries, we should merge them all
# # into one op to make it easier to manage
# summary_op = tf.summary.merge_all()


with tf.Session(config=config) as sess:
   
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs): 	
        print("Epoch: ",i)
        sess.run(train_init)	# drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += tf.reduce_sum(l)
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, sess.run(total_loss)/n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(test_init)			# drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds/n_test))
writer.close()

