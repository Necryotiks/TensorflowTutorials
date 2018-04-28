import tensorflow as tf
shaloam = tf.constant('Shaloam, Tensorflow!')
session = tf.Session()
print(session.run(shaloam))
