import tensorflow as tf
import os
from tensorflow.contrib.layers import flatten
import numpy as np
import tensorflow.contrib.slim as slim
from Ldata_helper import *


def model1(input, is_training, keep_prob):
    input = tf.reshape(input, [-1, 28,28,1])
    batch_norm_params = {'decay': 0.95, 'updates_collections':None}
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                       normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, activation_fn=tf.nn.crelu):
            print(input.get_shape())
            conv1 = slim.conv2d(input, 16, 5, scope='conv1') #output channels equal to number of kernels
            print(conv1.get_shape())
            pool1 = slim.max_pool2d(conv1, 2, scope='pool1') #output channels equal to conv1 channelss
            print(pool1.get_shape())
            conv2 = slim.conv2d(pool1, 32, 5, scope='conv2')
            print(conv2.get_shape())
            pool2 = slim.max_pool2d(conv2, 2, scope='pool2')
            print(pool2.get_shape())
            flatten = slim.flatten(pool2)
            print(flatten.get_shape())
            fc = slim.fully_connected(flatten, 1024, scope='fc1')
            print(fc.get_shape())
            """
            (?, 28, 28, 1)
            (?, 28, 28, 32)
            (?, 14, 14, 32)
            (?, 14, 14, 64)
            (?, 7, 7, 64)
            (?, 3136)
            (?, 2048)
            """
            drop = slim.dropout(fc, keep_prob=keep_prob)
            logits = slim.fully_connected(drop, 10, activation_fn=None, scope='logits')
            return logits


def train(model, model_path, train_log_path, test_log_path):
    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(dtype=tf.float32, shape=[None,28*28])
        Y = tf.placeholder(dtype=tf.float32,shape=[None,10])
        is_training = tf.placeholder(dtype=tf.bool)
        logit = model(X, is_training, .7)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=Y))
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit,1), tf.arg_max(Y,1)), tf.float32))
        global_step = tf.Variable(0, trainable=False)

        #decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        lr = tf.train.exponential_decay(0.1, global_step, 1000, 0.95, staircase=True)
        optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
        update = slim.learning.create_train_op(loss, optimizer, global_step)
        mnist = read_mnist()
        saver = tf.train.Saver()

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('acc', acc)
        merged_summary_op = tf.summary.merge_all()
        train_summary_writter = tf.summary.FileWriter(train_log_path, graph=tf.get_default_graph())
        test_summary_writter = tf.summary.FileWriter(test_log_path, graph=tf.get_default_graph())

        init = tf.global_variables_initializer()
        iter_num = 2000
        batch_size = 256

        with tf.Session() as sess:
            sess.run(init)
            try:
                saver.restore(sess, model_path)
            except:
                pass
            for i in range(iter_num):
                x,y = mnist.train.next_batch(batch_size)
                sess.run(update, feed_dict={X:x, Y:y, is_training: True})
                if i % 100 == 0:
                    x_test, y_test = mnist.test.next_batch(batch_size)
                    print('train:', sess.run(acc, feed_dict={X:x,Y:y,is_training:False}))
                    print('test:', sess.run(acc, feed_dict={X:x_test,Y:y_test,is_training:False}))
                    saver.save(sess, model_path)

                    g, summary = sess.run([global_step, merged_summary_op], feed_dict={X:x,Y:y,is_training:False})
                    train_summary_writter.add_summary(summary, g)
                    train_summary_writter.flush()

                    g, summary = sess.run([global_step, merged_summary_op], feed_dict={X:x_test,Y:y_test,is_training:False})
                    test_summary_writter.add_summary(summary, g)
                    test_summary_writter.flush()

            test_summary_writter.close()
            train_summary_writter.close()


if __name__=='__main__':
    from Lglobal_defs import *
    path_mnist_train = mk_dir(join(PATH_SAVING, 'mnist_train'))
    train(model1, join(path_mnist_train, 'model'), join(path_mnist_train, 'train_log'), join(path_mnist_train, 'test_log'))