import sys
import numpy as np
import tensorflow as tf
import time
import util_tensorflow

import os
import pickle


from options import get_options
from data_utils import DataManager






class MLP(object):
  '''some thing wrong, don't use this'''
  def __init__(self,
               learning_rate=0.005,
               layer_size= [3072,100,10],
               regula_lambda=0.6,
               is_batchnorm=False,
               is_training = True,
               activation_fn=tf.nn.sigmoid,
               epsilon=1e-8
               ):
    self.learning_rate = learning_rate
    self.layer_size = layer_size
    self.activation_fn = activation_fn
    self.is_batchnorm = is_batchnorm
    self.is_training = is_training
    self.epsilon = epsilon
    self.regularizer = tf.contrib.layers.l2_regularizer(regula_lambda)
    # why should we use He initialization when activation function is relu
    # and xavier initialization when function is not relu
    self._get_weights = util_tensorflow.get_he_weight \
      if 'elu' in activation_fn.__name__ else util_tensorflow.get_xavier_weights

    self._create_network()
    self._create_loss_optimizer()

  def _create_network(self):
    # tf Graph input
    self.images_placeholder = tf.placeholder(tf.float32, shape=[None, 3072],name='images_placeholder')
    self.labels_placeholder = tf.placeholder(tf.int64, shape=[None],name='labels_placeholder')

    with tf.variable_scope("MLP"):
      self.logits = self._create_layer(self.images_placeholder,reuse=False)


  def _create_layer(self, image, reuse):

    last_layer = image

    # Create hidden Layer
    with tf.variable_scope('Hidden_Layer',regularizer=self.regularizer,reuse=reuse):
      for i in range(1, len(self.layer_size)-1):
        layer_name = 'Layer_' + str(i)
        with tf.variable_scope(layer_name,regularizer=self.regularizer,reuse=reuse):
          W = self._get_weights((self.layer_size[i - 1],
                                 self.layer_size[i]))
          if self.is_batchnorm:
            current_layer = self.activation_fn(
                                util_tensorflow.batch_norm(
                                tf.matmul(last_layer, W),self.is_training))
          else:
            b = util_tensorflow.get_bias(self.layer_size[i])
            current_layer = self.activation_fn(
                                tf.matmul(last_layer, W)+b)
          last_layer = current_layer

    # Create Output Layer
    with tf.variable_scope('Output_Layer',regularizer=self.regularizer,reuse=reuse):
      W = util_tensorflow.get_he_weight((self.layer_size[-2],
                                          self.layer_size[-1]))
      b = util_tensorflow.get_bias(self.layer_size[-1])
      logits = tf.matmul(last_layer, W)+b

    return logits

  def _create_loss_optimizer(self):

    with tf.variable_scope('CostFunction'):
      self.class_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.logits,labels=self.labels_placeholder))
      self.regular_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
      self.loss = self.class_loss + self.regular_loss

    # Operation comparing prediction with true label
    correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.labels_placeholder)

    # Operation calculating the accuracy of our predictions
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')

    loss_summary_op = tf.summary.scalar('mlp_loss', self.loss)
    accuracy_summary_op = tf.summary.scalar('accuracy', self.accuracy)
    self.summary_op = tf.summary.merge([loss_summary_op,accuracy_summary_op])

    self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="MLP")

    self.optimizer = tf.train.AdamOptimizer(
      learning_rate=self.learning_rate,
      epsilon=self.epsilon).minimize(self.loss, var_list=self.variables)

  def get_accuracy(self, sess, image, label):
    # it will be crushed when the size of image is to large!!
    accuracy,loss = sess.run((self.accuracy,self.loss)
                      ,feed_dict={self.images_placeholder:image,
                                  self.labels_placeholder:label})
    return accuracy,loss

  def get_vars(self):
    return self.variables

  def partial_fit(self, sess, summary_writer, step, image, label):
    '''Train model based on mini-batch of input data'''
    _, loss, accuracy, summary_str = sess.run((self.optimizer,
                                    self.loss,
                                    self.accuracy,
                                    self.summary_op),
                                   feed_dict={self.images_placeholder:image,
                                              self.labels_placeholder:label})
    summary_writer.add_summary(summary_str, step)
    return loss, accuracy

class k_Nearest_Neighbour(object):
    ''' a kNN classifier with L2 distance '''
    def __init__(self):
        pass
    def fit(self,X,y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X[0:5000]
        self.y_train = y[0:5000]
    def perdict(self, X, k = 1):

      dists = self.compute_distances(X)
      num_test = dists.shape[0]
      y_pred = np.zeros(num_test)
      for i in range(num_test):
        closest_y = []
        closest_y = self.y_train[np.argsort(dists[i])[:k]]
        y_pred[i] = np.argmax(np.bincount(closest_y))
      return y_pred

    def compute_distances(self, X):
      """
      Compute the distance between each test point in X and each training point
      in self.X_train using no explicit loops.
      Input / Output: Same as compute_distances_two_loops
      """
      num_test = X.shape[0]
      num_train = self.X_train.shape[0]
      dists = np.zeros((num_test, num_train))

      # pass
      dists = np.sqrt(-2 * np.dot(X, self.X_train.T) + np.sum(np.square(self.X_train), axis=1) + np.transpose(
        [np.sum(np.square(X), axis=1)]))

      return dists
    def get_accuracy(self, X, Y, k = 1):

      X = X[0:1000]
      Y = Y[0:1000]
      pred = self.perdict(X,k)
      num_correct = np.sum(pred==Y)
      accuracy = float(num_correct)/ len(Y)
      return  accuracy




