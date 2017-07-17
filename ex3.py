

# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""

#best so far
#learning rate 0.001 epochs 150  hidden 1  256  hidden 2  70  hidden 3  30

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist
import scipy


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)




learning_rate = 0.001
training_epochs = 100
batch_size = 128
display_step = 1
examples_to_show = 15


"""
# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_hidden_1 = 600 # 1st layer num features
n_hidden_2 = 300 # 2nd layer num features
n_hidden_3 = 200
n_hidden_4 = 100
n_hidden_5 = 100

#n_hidden_2 = 30
"""


dims = [784,1024,256,64,30]

#print ("learning rate", learning_rate, "epochs", training_epochs, " hidden 1 ", n_hidden_1, " hidden 2 ", n_hidden_2, " hidden 3 ", n_hidden_3, "keep prob 0.8" )


# tf Graph input (only pictures)
X = tf.placeholder("float", [None, dims[0]])

keep_prob = tf.placeholder("float", None)

dim_pairs = zip(dims, dims[1:])

weights = {"encoder_h%d"%(i+1) : tf.Variable(tf.truncated_normal(dim_pair, stddev=0.1)) for i,dim_pair in enumerate(dim_pairs)}
weights.update({"decoder_h%d"%(i+1) : tf.Variable(tf.truncated_normal(dim_pair[::-1], stddev=0.1)) for i,dim_pair in enumerate(dim_pairs[::-1])})
biases = {"encoder_b%d"%(i): tf.Variable(tf.constant(0.1,shape = [dims[i]])) for i in range(1,len(dims))}
biases.update({"decoder_b%d"%(i): tf.Variable(tf.constant(0.1,shape = [dims[-i-1]])) for i in range(1,len(dims))[::-1]})


def masking_noise(X, v):
    """ Apply masking noise to data in X, in other words a fraction v of elements of X
    (chosen at random) is forced to zero.
    :param X: array_like, Input data
    :param v: int, fraction of elements to distort
    :return: transformed data
    """
    X_noise = X.copy()

    n_samples = X.shape[0]
    n_features = X.shape[1]

    for i in range(n_samples):
        mask = np.random.randint(0, n_features, v)

        for m in mask:
            X_noise[i][m] = 0.

    return X_noise


def encoder_2(x):
    #global keep_prob
    global dims
    layers = [x]
    dropouts = []
    print (list(enumerate(dims)))
    print (biases.keys())
    #exit()
    for i,dim in list(enumerate(dims))[:-1]:
        if i==0:
            dropouts.append(layers[-1])
            #dropouts.append(tf.nn.dropout(layers[-1], keep_prob = keep_prob))
        else:
            #k = max(keep_prob, 0.9)
            dropouts.append(tf.nn.dropout(layers[-1], keep_prob=keep_prob))
        try:
            layers.append(tf.nn.relu(tf.add(tf.matmul(dropouts[-1], weights["encoder_h%d"%(i+1)]), biases["encoder_b%d"%(i+1)])))
        except:
            print ("fuck that ")

    return layers[-1]

"""
# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))

    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))

    layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['decoder_h4']), biases['decoder_b4']))

    layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4, weights['decoder_h5']), biases['decoder_b5']))

    return layer_5
"""

def decoder_2(x):
    global dims
    layers = [x]
    for i in range(1,len(dims)):
        layers.append(tf.nn.relu(tf.add(tf.matmul(layers[-1], weights["decoder_h%d"%(i)]), biases["decoder_b%d"%(i)])))
        #layers.append(tf.nn.sigmoid(tf.add(tf.matmul(layers[-1], weights["decoder_h%d" % (i)]), biases["decoder_b%d" % (i)])))
    return layers[-1]


def check_model(data, encoder, labels):
    import scipy
    import numpy as np
    global sess
    encoded_data = sess.run(encoder, feed_dict={X: data, keep_prob: 1.0})
    distance_matrix = scipy.spatial.distance.squareform(pdist(encoded_data))
    d_m_2 = distance_matrix[:, :]
    np.fill_diagonal(d_m_2, np.inf)
    labels = np.argmax(labels, 1)  # these are the labels!
    #print (labels[:20])
    predicate = labels[np.argmin(d_m_2, 1)]  # get the indecies of the closest data sample
    #print (data[1,:])
    print ("correct: ", np.sum(labels == predicate), labels.shape, predicate.shape)

# Construct model
encoder_op = encoder_2(X)
decoder_op = decoder_2(encoder_op)

# Prediction
y_pred = decoder_op
x_encode = encoder_op

# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

#optimizer1 = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

import random
l = range(len(mnist.train.labels))
# Launch the graph
with tf.Session() as sess:
    import matplotlib
    import random
    sess.run(init)
    print("num examples are ", mnist.train.num_examples, mnist.validation.num_examples, mnist.test.num_examples)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        r = 0.2#random.uniform(0.2, 0.5)
        print ('this is r ', r)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size, shuffle=False)
            # Run optimization op (backprop) and cost op (to get loss value)
            #print (r)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, keep_prob:r})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))
        #check_model(mnist.test.images, x_encode, mnist.test.labels)
        print("size", len(mnist.train.images), len(mnist.train.labels))
        #exit()
        check_model(mnist.train.images[:1000], x_encode, mnist.train.labels[:1000])
        check_model(mnist.test.images, x_encode, mnist.test.labels)
        #print ("size", len(mnist.test.labels), len(mnist.test.labels))
    print("Optimization Finished!")




    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show], keep_prob:1.0})

    plt.imshow(encode_decode[0].reshape(28,28))
    plt.gray()
    plt.show()
    exit()
    encoded_data = sess.run(x_encode, feed_dict={X: mnist.test.images, keep_prob:1.0})
    #check_model(mnist.test.images, x_encode, mnist.test.labels)
exit()
distance_matrix = scipy.spatial.distance.squareform(pdist(encoded_data))

d_m_2 = distance_matrix[:,:]
np.fill_diagonal(d_m_2,np.inf)

labels = np.argmax(mnist.test.labels,1) #these are the labels!
predicate = labels[np.argmin(d_m_2,1)] #get the indecies of the closest data sample
print ("this is the ammount of coorect clasificcations in the test set", np.sum(labels==predicate)) #count how many similar values are there!

#check_model(mnist.test.images,x_encode,mnist.test.labels)

"""
def check_model(data, encoder, labels):
    import scipy
    import numpy as np 
    encoded_data = sess.run(encoder, feed_dict={X: data, keep_prob: 1.0})
    distance_matrix = scipy.spatial.distance.squareform(pdist(encoded_data))
    d_m_2 = distance_matrix[:, :]
    np.fill_diagonal(d_m_2, np.inf)
    labels = np.argmax(labels, 1)  # these are the labels!
    predicate = labels[np.argmin(d_m_2, 1)]  # get the indecies of the closest data sample
    print ("correct: ", np.sum(labels == predicate))
"""

