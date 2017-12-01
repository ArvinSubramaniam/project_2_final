"""Creating the CNN Model + Architecture"""

import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image

import code
import tensorflow.python.platform

import numpy
import tensorflow as tf
from support_functions import *

NUM_CHANNELS = 3 # RGB images
NUM_LABELS = 2
SEED = 66478  # Set to None for random seed.

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16 

def model(data,train=False):
    """The Model definition. Returns output layer after linear activation and variables.
    Variables returned defined before using the data, hence unaffected by it."""
    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 512],
                            stddev=0.1,
                            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))    
        
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME')

    conv2 = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    pool2 = tf.nn.max_pool(relu2,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

    # Uncomment these lines to check the size of each layer
    # print 'data ' + str(data.get_shape())
    # print 'conv ' + str(conv.get_shape())
    # print 'relu ' + str(relu.get_shape())
    # print 'pool ' + str(pool.get_shape())
    # print 'pool2 ' + str(pool2.get_shape())
    
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool2.get_shape().as_list()
    reshape = tf.reshape(
        pool2,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    out = tf.matmul(hidden, fc2_weights) + fc2_biases

    if train == True:
        summary_id = '_0'
        s_data = get_image_summary(data)
        filter_summary0 = tf.summary.image('summary_data' + summary_id, s_data)
        s_conv = get_image_summary(conv)
        filter_summary2 = tf.summary.image('summary_conv' + summary_id, s_conv)
        s_pool = get_image_summary(pool)
        filter_summary3 = tf.summary.image('summary_pool' + summary_id, s_pool)
        s_conv2 = get_image_summary(conv2)
        filter_summary4 = tf.summary.image('summary_conv2' + summary_id, s_conv2)
        s_pool2 = get_image_summary(pool2)
        filter_summary5 = tf.summary.image('summary_pool2' + summary_id, s_pool2)
    
    return out, conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights, fc2_biases