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
	fname = 'cifar-10-batches-py/data_batch_'+str(i)
	print 'Loading : ', fname
        raw_data = unpickle( fname )
        X.append(np.array( raw_data['data'] ).astype('float32'))
        Y.append(np.array( raw_data['labels']).astype('float32'))

    X = np.concatenate( X )
    Y = np.concatenate( Y )

    Y_hot = np.zeros( (Y.shape[0],10))
    for i in range(Y.shape[0]):
        Y_hot[ i, int(Y[i]) ] = 1.0

    return X, Y_hot


X, Y_hot = load_raw()


# Show a few images
for i in range(10):
    im = X[i,:].reshape( (-1,3,32,32) ).transpose(0,2,3,1)  # RGB Nx28x28x3
    label_indx = Y_hot[i,:].argmax()
    print 'Label : ', label_indx, meta[label_indx]
    cv2.imshow( 'win', cv2.cvtColor( im[0,:,:,:].astype('uint8'), cv2.COLOR_RGB2BGR ))
    cv2.waitKey(0)
