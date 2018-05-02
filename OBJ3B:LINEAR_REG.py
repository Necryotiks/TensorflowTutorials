import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
DATA_FILE = "datasets/birthrate.txt"

# Step 1: read in data from the .txt file
# data is a numpy array of shape (190, 2), each row is a datapoint
#TODO: fix this
data= np.loadtxt(DATA_FILE,skiprows=1,usecols=(1,2)).astype(np.float32)
# dataset = tf.data.Dataset.from_tensor_slices((data[:,0],data[:,1]))
# Step 2: create placeholders for X (birth rate) and Y (life expectancy)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')
c = tf.constant('Example 1')

# Step 3: create weight and bias, initialized to 0
w = tf.get_variable('weights1', initializer=tf.constant(0.0))
# u = tf.get_variable('weights2', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

# Step 4: construct model to predict Y (life expectancy from birth rate)
# Y_predicted = ((w *X) + b) 
Y_predicted = tf.add(tf.multiply(w,X),b)
# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')
def huber_loss(labels, predictions, delta=14.0):
    residual = tf.abs(labels - predictions)
    def f1(): return 0.5 * tf.square(residual)
    def f2(): return delta * residual - 0.5 * tf.square(delta)
    return tf.cond(residual < delta, f1, f2)

#loss = huber_loss(Y,Y_predicted)
# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    # iterator = dataset.make_initializable_iterator()
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))
    # X,Y= iterator.get_next()


      # Step 8: train the model
    for i in range(1000): # run 100 epochs
            # Session runs train_op to minimize loss
        for x,y in data:
            sess.run(optimizer, feed_dict={X:x,Y:y})
	# sess.run(iterator.initializer)
	# try:
		# while True:
                            # sess.run([optimizer]) 
	# except tf.errors.OutOfRangeError:
		# pass
    # Step 9: output the values of w and b
    w_out, b_out = sess.run([w, b])
    print(w_out,b_out)
    plt.plot(data[:,0], data[:,1], 'bo', label='Testing data')
    plt.plot(data[:,0],w_out*data[:,0]+b_out, label='line')
    plt.legend()
    plt.show()
