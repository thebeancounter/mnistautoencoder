import idx2numpy
import cPickle, gzip, numpy

import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
#sess = tf.InteractiveSession()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print mnist


exit()


node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)


sess = tf.Session()
print(sess.run([node1, node2]))


node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))



exit()
# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_set, train_tags = train_set
val_set, val_tags = valid_set
test_set, test_tags = test_set

