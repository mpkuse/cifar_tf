import tensorflow as tf

import numpy as np
import cv2
import matplotlib.pyplot as plt


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

    # NORMPROP
    # return (tf.nn.relu(x) - 0.039894228) / 0.58381937
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    pool_out = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
    # NORMPROP
    # return (pool_out - 1.4850) / 0.7010
    return pool_out





def make_alexnet( x ):
    with tf.device( '/cpu:0'):
        wc1 = tf.get_variable( 'wc1', [5,5,3,64] )#, initializer=tf.random_normal_initializer() )
        wc2 = tf.get_variable( 'wc2', [5,5,64,64] )#, initializer=tf.random_normal_initializer())
        wd1 = tf.get_variable( 'wd1', [4096,1024] )#, initializer=tf.random_normal_initializer())
        wd2 = tf.get_variable( 'wd2', [1024,10] )#, initializer=tf.random_normal_initializer())

        bc1 = tf.get_variable( 'bc1', [64] )#, initializer=tf.constant_initializer(0.01) )
        bc2 = tf.get_variable( 'bc2', [64] )#, initializer=tf.constant_initializer(0.01))
        bd1 = tf.get_variable( 'bd1', [1024] )#, initializer=tf.constant_initializer(0.01))
        bd2 = tf.get_variable( 'bd2', [10] )#, initializer=tf.constant_initializer(0.01))



    conv1 = conv2d( x, wc1, bc1)

    conv1 = maxpool2d( conv1, k=2 )

    conv2 = conv2d( conv1, wc2, bc2 )
    conv2 = maxpool2d( conv2, k=2 )

    # reshape
    fc1 = tf.reshape( conv2, [-1,4096])


    fc1 = tf.matmul( fc1, wd1 ) + bd1
    fc1 = tf.nn.relu( fc1 )

    out = tf.add( tf.matmul( fc1, wd2 ) , bd2, name='tower_loss' )

    return out


def report_means( x ):
    mean,variance = tf.nn.moments( x, [0,1,2] )
    print 'mean : ', mean
    print 'var : ', variance
    return mean, variance




X, Y_hot = load_raw()


# Make the gradient descent on CPU
#with tf.Graph().as_default(), tf.device('/cpu:0'):
with tf.device( '/cpu:0'):
    # Make Optimizer
    opt = tf.train.AdamOptimizer(0.001)

    with tf.variable_scope( 'trainable_vars', reuse=None ) as scope:
        # Make trainable variables
        wc1 = tf.get_variable( 'wc1', [5,5,3,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())#tf.random_normal_initializer() )
        wc2 = tf.get_variable( 'wc2', [5,5,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())#tf.random_normal_initializer())
        wd1 = tf.get_variable( 'wd1', [4096,1024], initializer=tf.contrib.layers.xavier_initializer())#tf.random_normal_initializer())
        wd2 = tf.get_variable( 'wd2', [1024,10], initializer=tf.contrib.layers.xavier_initializer())#tf.random_normal_initializer())

        bc1 = tf.get_variable( 'bc1', [64], initializer=tf.constant_initializer(0.01) )
        bc2 = tf.get_variable( 'bc2', [64], initializer=tf.constant_initializer(0.01))
        bd1 = tf.get_variable( 'bd1', [1024], initializer=tf.constant_initializer(0.01))
        bd2 = tf.get_variable( 'bd2', [10], initializer=tf.constant_initializer(0.01))


        # with tf.variable_scope( 'bn', reuse=None ):
        # beta = tf.get_variable( 'beta', [3], initializer=tf.constant_initializer(0.01)  )
            # gamma = tf.get_variable( 'gamma', [3], initializer=tf.constant_initializer(value=1) )


        print 'current variable scope : ', tf.get_variable_scope()

    # tf.get_variable_scope().reuse_variables()




tower_grad = [] #gradients of each tower
placeholder_x = []
placeholder_y = []
tower_cost = []
tower_infer = []
tower_batch_mean = []
tower_batch_var = []
for gpu_id in [0,1]:
    with tf.device( '/gpu:'+str(gpu_id) ):
        with tf.name_scope( 'tower_'+str(gpu_id)), tf.variable_scope( 'trainable_vars', reuse=True ):

            #place holder
            x = tf.placeholder( 'float', [None,32,32,3], name='x' )
            y = tf.placeholder( 'float', [None,10], name='y' )
            placeholder_x.append(x)
            placeholder_y.append(y)
            print x.name

            batch_pred = make_alexnet( x )
            cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(batch_pred,y) )

            mean_x, variance_x = report_means( x )
            tower_batch_mean.append( mean_x )
            tower_batch_var.append( variance_x )

            tower_cost.append(cost)
            tower_infer.append( batch_pred )





            #all_vars = tf.trainable_variables()
            #print 'scope : ', scope.name

            # Variable `grads` contain a list containing 2 elements each
            # ie. ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = opt.compute_gradients( cost  )


            tower_grad.append( grads )


            print batch_pred.name





# Average the gradients in tower. Basically add tower_1's gradients into tower_0's
# gradient
print '###'
avg_grad = []
for i in range(len(tower_grad[0])):
    t0_grad = tower_grad[0][i][0]
    t0_var  = tower_grad[0][i][1]
    t1_grad = tower_grad[1][i][0]

    mean_grad = tf.mul( tf.add(t0_grad,t1_grad), tf.convert_to_tensor(0.5) )

    avg_grad.append( (mean_grad, t0_var) )
    print tower_grad[0][i][1].name
print '###'

print '---trainable_variables---'
for tr_vars in tf.trainable_variables():
    print tr_vars.name
print '-END--trainable_variables--END-'




with tf.device( '/cpu:0'):
    apply_gradient_op = opt.apply_gradients( avg_grad )

init = tf.initialize_all_variables()




# quit()

## Session. Note: allow_soft_placement is a hacky way of going around a tensorflow bug which does not
# let computation of tf.nn.moments() on a gpu
session = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
session.run( init )

tf.train.start_queue_runners(sess=session)

## Tensorboard
with tf.device( '/cpu:0' ):
    tf.scalar_summary( 'tower0_cost', tower_cost[0] )
    tf.scalar_summary( 'tower1_cost', tower_cost[1] )

summary_writer = tf.train.SummaryWriter('logs/xavier_init', graph=tf.get_default_graph())
summary_op = tf.merge_all_summaries()


loss_gpu0 = []
loss_gpu1 = []
for step in range(500): #Iterations
    r0 = np.random.randint( 0, X.shape[0], 256 )  #batch to set to gpu 0
    r1 = np.random.randint( 0, X.shape[0], 256 ) #batch to set to gpu1

    B0 = X[r0,:].reshape( (-1,3,32,32) ).transpose(0,2,3,1)
    la0 = Y_hot[r0,]
    B1 = X[r1,:].reshape( (-1,3,32,32) ).transpose(0,2,3,1)
    la1 = Y_hot[r1,]





    _,aa,bb,ss = session.run( [apply_gradient_op,tower_cost[0],tower_cost[1],summary_op], feed_dict={placeholder_x[0]:B0,placeholder_x[1]:B1,placeholder_y[0]:la0,placeholder_y[1]:la1} )
    bm1, bm2, bv1, bv2 = session.run( [tower_batch_mean[0], tower_batch_mean[1], tower_batch_var[0], tower_batch_var[1]], feed_dict={placeholder_x[0]:B0,placeholder_x[1]:B1,placeholder_y[0]:la0,placeholder_y[1]:la1})
    loss_gpu0.append(aa)
    loss_gpu1.append(bb)
    print '%-5d, %-0.4f, %-0.4f :: %0.4f,%0.4f,    %0.4f,%0.4f' %(step,aa,bb,   bm1[0], bv1[0], bm2[0], bv2[0])

    # Tensorboard
    if step%20 == 0:
        print 'summary_writer()'
        summary_writer.add_summary( ss , step)


## Saver
saver = tf.train.Saver()
fname = 'trained_models/no_norm.tfm'
print 'saving ', saver.save(session, fname )


## Model Evaluation
correct = 0
for i in range(10000):
    B0 = X[i:i+1,:].reshape( (-1,3,32,32) ).transpose(0,2,3,1)
    la0 = Y_hot[i:i+1,]
    b = session.run( tower_infer[0],feed_dict={placeholder_x[0]:B0,placeholder_y[0]:la0})
    print i,b.argmax(), la0.argmax()
    if b.argmax() == la0.argmax():
        correct = correct + 1

print 'correct : ',correct
