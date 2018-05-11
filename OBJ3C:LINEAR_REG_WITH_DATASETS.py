import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
DATA_FILE = "./datasets/birthrate.txt"
data = np.loadtxt(DATA_FILE,skiprows=1,usecols=(1,2)).astype(np.float32)
dataset = tf.data.Dataset.from_tensor_slices((data[:,0],data[:,1]))
# dataset = dataset.batch(128)
iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
train_init = iterator.make_initializer(dataset)#  if using dataset.repeat() then we can use a one shot iterator
X,Y = iterator.get_next()

w=tf.get_variable('slope',initializer=tf.constant(0.0))
b=tf.get_variable('bias',initializer=tf.constant(0.0))
y_pred= tf.add(tf.multiply(w,X),b)
loss = tf.reduce_mean(tf.square((Y-y_pred)))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    w_curr = 0
    b_curr = 0
    start = time.time()
    for i in range(400): #Epoch number
        sess.run(train_init) # not necessary if using a one shot iterator.
        while True:#loop through dataset
            try:
                sess.run(optimizer)
            except tf.errors.OutOfRangeError:
                 break
        w_curr, b_curr = sess.run([w,b])
        print(w_curr,b_curr)
    end = time.time()
    print("Total train time: ",end-start)
plt.plot(data[:,0], data[:,1], 'bo', label='Testing data')
plt.plot(data[:,0],w_curr*data[:,0]+b_curr, label='line')
plt.legend()
plt.show()
