import tensorflow as tf
import math
import time
import numpy as np
import os
import sys
import tf_util

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl

def extract_features(point_cloud, is_training, bn_decay=None):
    """ ConvNet baseline, input is BxNx3 gray image """
    batch_size = point_cloud.get_shape()[0].value
    print("Batch Size {}".format(batch_size))

    num_point = point_cloud.get_shape()[1].value
    print("Num points {}".format(num_point))

    input_image = tf.expand_dims(point_cloud, -1)
    print("Input image {}".format(input_image))

    # CONV
    net = tf_util.conv2d(input_image, 64, [1,3], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)

    print("The output of the conv1 layer is {}".format(net))

    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)

    print("The output of the conv2 layer is {}".format(net))

    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)

    print("The output of the first conv3 layer is {}".format(net))

    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)

    print("The output of the first conv4 layer is {}".format(net))

    #net = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
     #                     bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)

    #print("The output of the first conv5 layer is {}".format(net))

    return net

if __name__ == "__main__":
    with tf.Graph().as_default():
        a = tf.placeholder(tf.float32, shape=(16,4096,3))
        net = get_model(a, tf.constant(True))
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.run(net, feed_dict={a:np.random.rand(16,4096,3)})
            
