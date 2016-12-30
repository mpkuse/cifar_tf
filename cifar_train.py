import tensorflow as tf

import numpy as np
import cv2
import matplotlib.pyplot as plt


meta = [ 'airplane',\
'automobile',\										
'bird',\										
'cat',\										
'deer',\										
'dog',	\									
'frog',\										
'horse',\										
'ship',	\									
'truck' ]

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def load_raw():
    X = []
    Y = []
    for i in range(1,6):
        raw_data = unpickle( 'cifar-10-batches-py/data_batch_'+str(i) )
        X.append(np.array( raw_data['data'] ).astype('float32'))
        Y.append(np.array( raw_data['labels']).astype('float32'))

    X = np.concatenate( X )
    Y = np.concatenate( Y )

    Y_hot = np.zeros( (Y.shape[0],10))
    for i in range(Y.shape[0]):
        Y_hot[ i, int(Y[i]) ] = 1.0

    return X, Y_hot

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def make_alexnet( x, weights, biases):
    conv1 = conv2d( x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d( conv1, k=2 )

    conv2 = conv2d( conv1, weights['wc2'], biases['bc2'] )
    conv2 = maxpool2d( conv2, k=2 )

    # reshape
    fc1 = tf.reshape( conv2, [-1,4096])
    fc1 = tf.matmul( fc1,  weights['wd1'] ) + biases['bd1']
    fc1 = tf.nn.relu( fc1 )

    out = tf.add( tf.matmul( fc1, weights['wd2'] ) , biases['bd2'] )


    print out
    print fc1
    print conv1
    print conv2
    return out


def regularizer(weights, biases, lambda_val=0.01):
    w_decay = tf.nn.l2_loss(weights['wc1']) + tf.nn.l2_loss(weights['wc2']) + tf.nn.l2_loss(weights['wd1']) + tf.nn.l2_loss(weights['wd2'])
    b_decay = tf.nn.l2_loss(biases['bc1']) + tf.nn.l2_loss(biases['bc2']) + tf.nn.l2_loss(biases['bd1']) + tf.nn.l2_loss(biases['bd2'])

    return tf.mul( lambda_val, tf.add(w_decay,b_decay) )





X, Y_hot = load_raw()




weights = {
    #5x5 conv 3 input, 64 output
    'wc1': tf.Variable( tf.random_normal([5,5,3,64])),

    #5x5 conv 32 input, 64 output
    'wc2': tf.Variable( tf.random_normal([5,5,64,64])),

    #fc, input 8*8*64 ie 4096; output dim is 1024
    'wd1': tf.Variable( tf.random_normal([4096,1024])),

    'wd2': tf.Variable( tf.random_normal([1024,10]))
}

biases = {
    'bc1': tf.Variable( tf.random_normal([64])),
    'bc2': tf.Variable( tf.random_normal([64])),
    'bd1': tf.Variable( tf.random_normal([1024])),
    'bd2': tf.Variable( tf.random_normal([10]))
}


x = tf.placeholder( 'float', [None,32,32,3] )
y = tf.placeholder( 'float', [None,10] )


with tf.device( '/gpu:1'):
    pred = make_alexnet(x, weights, biases)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(pred,y) )  + regularizer(weights, biases, lambda_val=0.01)

    train_op = tf.train.AdamOptimizer( 0.001).minimize(cost)


session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
session.run( tf.initialize_all_variables() )


## Tensorboard
with tf.device( '/cpu:0' ):
    tf.scalar_summary( 'cost', cost )

summary_writer = tf.train.SummaryWriter('logs/', graph=tf.get_default_graph())
summary_op = tf.merge_all_summaries()


loss_ary = []
for i in range(20000):
    r = np.random.randint( 0, X.shape[0], 512 )
    im = X[r,:].reshape( (-1,3,32,32) ).transpose(0,2,3,1)  # RGB Nx28x28x3
    a,b,c = session.run( [train_op,cost,summary_op], feed_dict={x:im, y:Y_hot[r,] } )
    print i, b
    loss_ary.append(b)

    if i%10 == 0:
        summary_writer.add_summary( c, i)




plt.plot( loss_ary )
plt.show()


# Evaluation
correct = 0
for i in range(10000):
    im = X[i,:].reshape( (-1,3,32,32) ).transpose(0,2,3,1)  # RGB Nx28x28x3
    b = session.run( pred, feed_dict={x:im})
    print i,b.argmax(), Y_hot[i,:].argmax()
    if b.argmax() == Y_hot[i,:].argmax():
        correct = correct + 1

print 'correct : ',correct
