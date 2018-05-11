import tensorflow as tf
#This object will go over the basics or Tensorflow arithmetic oporations

#Definitions:
#tf.Session(): A Session object encapsulates the environment in which Operation objects are executed and Tensor objects are evaluated.
#Session will also allocate memory to store the current values or variables

#Example 1
print("Example 1:")
a = tf.add(3,5) #NOTE: This op has been queued and will not be executed until session.run() is called
session = tf.Session()
print(session.run(a)) #Should print 8
session.close() #Closes the running session.
print('\n')

#Example 2
print("Example 2:")
a = tf.add(2,2)
with tf.Session() as session: #NOTE: This saves you from having to assign and then later close the session(i.e x = tf.Session() -> x.close())
    print(session.run(a)) # Should print 4  
print('\n')

#Example 3
print("Example 3:")
x = 5
y = 2
op1 = tf.add(x,y)
op2 = tf.multiply(x,y)
op3 = tf.pow(op1,op2) # This will compute the necessary dependencies when op3 is executed.
with tf.Session() as sess:
    print(sess.run(op3)) #Should print 7^10
print('\n')

print("Example 4:")
x = 5
y = 2
op1 = tf.add(x,y)
op2 = tf.multiply(x,y)
noop = tf.constant('I do nothing!') #This won't do shit because session only computes direct dependencies.
op3 = tf.pow(op1,op2) 
with tf.Session() as sess:
    print(sess.run(op3))
    print(noop) #This will only print out the Tensor type but not actually compute anything
print('\n')
