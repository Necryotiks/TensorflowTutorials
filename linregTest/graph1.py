import tensorflow as tf
import numpy as np
import time

datafile = 'data/kc_house_data.csv'
datafeat = np.loadtxt(datafile, skiprows=1, usecols=range(1,20),delimiter=',').astype(np.float32)
datalab = np.loadtxt(datafile, skiprows=1, usecols=20,delimiter=',').astype(np.float32)
datafeat = tf.convert_to_tensor(datafeat)
datafeat = tf.expand_dims(datafeat, -1)
datafeat = tf.transpose(datafeat, perm=[0,2,1])
dataset1 = tf.data.Dataset.from_tensor_slices(datafeat)
dataset2 = tf.data.Dataset.from_tensor_slices(datalab)
featin = dataset1.make_initializable_iterator()
labin = dataset2.make_initializable_iterator()
X = featin.get_next()
Y = labin.get_next()
w = tf.get_variable('weight', initializer=tf.random_normal(shape=[19,1]))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

y_pred=tf.add(tf.matmul(w,X),b)
loss = tf.reduce_mean(tf.square(Y-y_pred))

opt = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    w_curr = 0
    b_curr = 0
    totalloss = 0
    start = time.time()
    for i in range(3000):
        sess.run(featin.initializer)
        sess.run(labin.initializer)
        while True:
            try:
                _,l = sess.run([opt,loss])
                totalloss = l
            except tf.errors.OutOfRangeError:
                break
        print("Loss: ",totalloss," Epoch: ",i)
    end = time.time()
    print("Total train time: ", end-start)
#plt.plot(data[:,0],data[:,1], 'bo', label='Testing data')
#plt.plot(data[:,0],w_curr=data[:,0]+b_curr, label='line')
#plt.legend()
#plt.show()


