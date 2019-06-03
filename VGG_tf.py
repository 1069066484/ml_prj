import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def vgg19(inputs):
    batch_norm_params = {'decay': 0.99, 'updates_collections':None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        normalizer_fn=slim.batch_norm, 
                        normalizer_params=batch_norm_params):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.flatten(net, scope='flat')

    net = slim.fully_connected(net, 4096, scope='fc1')
    net = slim.dropout(net, keep_prob=0.5, scope='dropt1')
    net = slim.fully_connected(net, 4096, scope='fc2')
    net = slim.dropout(net, keep_prob=0.5, scope='dropt2')
    net = slim.fully_connected(net, 1000, scope='fc3')
    net = slim.softmax(net, scope='net')
    return net


def train(model, path_model, path_train_log, path_test_log):
    pass


if __name__=='__main__':
    pass