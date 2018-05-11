import tensorflow as tf
#Example 1
str1 = tf.constant('Example 1')
x = tf.constant(2,name="x")
y = tf.constant(8,name="y")
op1 = tf.add(x,y,name="Add_op")
writer = tf.summary.FileWriter('logdir',tf.get_default_graph())
with tf.Session() as session:
    print(session.run(str1))
    print(session.run(op1))
writer.close()



#tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)

# constant of 1d tensor (vector)
#a = tf.constant([2, 2], name="vector")

# constant of 2x2 tensor (matrix)
#b = tf.constant([[0, 1], [2, 3]], name="matrix")

#tf.zeros(shape, dtype=tf.float32, name=None)
# create a tensor of shape and all elements are zeros
#tf.zeros([2, 3], tf.int32) ==> [[0, 0, 0], [0, 0, 0]]

#tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)
# create a tensor of shape and type (unless type is specified) as the input_tensor but all elements are zeros.
# input_tensor [[0, 1], [2, 3], [4, 5]]
#tf.zeros_like(input_tensor) ==> [[0, 0], [0, 0], [0, 0]]

#tf.ones(shape, dtype=tf.float32, name=None)
# create a tensor of shape and all elements are ones
#tf.ones([2, 3], tf.int32) ==> [[1, 1, 1], [1, 1, 1]]
#tf.ones(shape, dtype=tf.float32, name=None)

# create a tensor of shape and all elements are ones
#tf.ones([2, 3], tf.int32) ==> [[1, 1, 1], [1, 1, 1]]
#tf.fill(dims, value, name=None)

# create a tensor filled with a scalar value.
#tf.fill([2, 3], 8) ==> [[8, 8, 8], [8, 8, 8]]

#tf.lin_space(start, stop, num, name=None)
# create a sequence of num evenly-spaced values are generated beginning at start. If num > 1, the values in the sequence increase by (stop - start) / (num - 1), so that the last one is exactly stop.
# comparable to but slightly different from numpy.linspace
#tf.lin_space(10.0, 13.0, 4, name="linspace") ==> [10.0 11.0 12.0 13.0]

#tf.range([start], limit=None, delta=1, dtype=None, name='range')
# create a sequence of numbers that begins at start and extends by increments of delta up to but not including limit
# slight different from range in Python

# 'start' is 3, 'limit' is 18, 'delta' is 3
#tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
# 'start' is 3, 'limit' is 1,  'delta' is -0.5
#tf.range(start, limit, delta) ==> [3, 2.5, 2, 1.5]
# 'limit' is 5
#tf.range(limit) ==> [0, 1, 2, 3, 4]

a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
c = tf.constant('Example 2')
with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(tf.div(b, a)))             #⇒ [[0 0] [1 1]], uses integer division
    print(sess.run(tf.divide(b, a)))          #⇒ [[0. 0.5] [1. 1.5]], uses floating point division

a = tf.constant([10, 20], name='a')
b = tf.constant([2, 3], name='b')
c = tf.constant('Example 3')
with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(tf.multiply(a, b)))           #⇒ [20 60] # element-wise multiplication
    print(sess.run(tf.tensordot(a, b, 1)))       #⇒ 80
    
s = tf.get_variable("scalar", initializer=tf.constant(2)) 
m = tf.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable("big_matrix", shape=(784, 10), initializer=tf.zeros_initializer())
op1 = tf.multiply(W,W)
c = tf.constant('Example 4')
with tf.Session() as sess:
    print(sess.run(c))
    sess.run(tf.global_variables_initializer()) #Initializes variables
    print(sess.run(tf.report_uninitialized_variables())) #Prints which variables are uninitialized.
    print(sess.run(op1))

#sess.run(tf.variables_initializer([a, b])) //initialize subset of variables
#sess.run(W.initializer) //initialize a single variable


V = tf.get_variable("normal_matrix", shape=(784, 10),initializer=tf.truncated_normal_initializer())

c = tf.constant('Example 5')
with tf.Session() as sess:
    print(sess.run(c))
    sess.run(tf.global_variables_initializer())
    print(sess.run(V))

c = tf.constant('Example 6')
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W.initializer)
sess2.run(W.initializer)
print(sess1.run(c))        # >> 20
print(sess1.run(W.assign_add(10)))        # >> 20
print(sess1.run(W.assign_add(10)))        # >> 20
print(sess2.run(W.assign_sub(2)))        # >> 8
print(sess1.run(W.assign_add(100)))        # >> 120
print(sess2.run(W.assign_sub(50)))        # >> -42
sess1.close()
sess2.close()

z = tf.constant('Example 7')
a = tf.placeholder(tf.float32, shape=[3]) # a is placeholder for a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)
c = a + b # use the placeholder as you would any tensor
with tf.Session() as sess:
    print(sess.run(z))
    # compute the value of c given the value of a is [1, 2, 3]
    print(sess.run(c, {a: [1, 2, 3]}))         # [6. 7. 8.]
    # feed_dict: is basically a dictionary with keys being the placeholders, value being the values of those placeholders.
    #### always separate the definition of ops and their execution when you can.
