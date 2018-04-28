#This is a very simple object designed to test if Tensorflow installed correctly. Nice and easy.
import tensorflow as tf
shaloam = tf.constant('Shaloam, Tensorflow!') #NOTE0: Declaring a variable/constant/whatever in TF is not done through '=' assignment operator.
session = tf.Session() #NOTE1: All models in TF require a session in order to execute them.
print(session.run(shaloam)) #"Executing" a graph.
