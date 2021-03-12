import os, sys
sys.path.append(os.getcwd())

from layers import Linear
from layers import LinearSN
from layers import Conv2D
from layers import Normalize
from layers import LayerNorm1d
from layers import ResidualLayer
from layers import ResidualLayer1d

import numpy as np
import tensorflow as tf

from functools import partial


def mnist_feature_learner(name, images, DIM, dropout=None, is_training=None):
    initializer = 'glorot_uniform'
    normalizer = 'BN'

    output = Conv2D('{}.feature.Conv.1'.format(name), images, DIM, initializer=initializer)
    output = tf.nn.max_pool(output, (1,1,2,2), (1,1,2,2), 'VALID', data_format='NCHW')

    output = Normalize('{}.feature.NORM.1'.format(name), output, method=normalizer, bn_is_training=is_training)
    output = tf.nn.relu(output)

    output = Conv2D('{}.feature.Conv.2'.format(name), output, 2*DIM, initializer=initializer)
    output = tf.nn.max_pool(output, (1,1,2,2), (1,1,2,2), 'VALID', data_format='NCHW')

    output = Normalize('{}.feature.NORM.2'.format(name), output, method=normalizer, bn_is_training=is_training)
    output = tf.nn.relu(output)

    output = Conv2D('{}.feature.Conv.3'.format(name), output, 4*DIM, initializer=initializer)
    output = tf.nn.max_pool(output, (1,1,2,2), (1,1,2,2), 'VALID', data_format='NCHW')

    output = Normalize('{}.feature.NORM.3'.format(name), output, method=normalizer, bn_is_training=is_training)
    output = tf.nn.relu(output)

    output = tf.reduce_mean(output, [2,3])
    return output


def predictor(predictor_name, representation, output_dim, params=None):
    output = Linear('{}.predictor.Linear'.format(predictor_name), representation, output_dim, initializer='glorot_uniform')
    return output


def partition_predictor(name, representation, DIM, is_training=None):
    nonlinearity = tf.nn.relu 
    initializer = 'glorot_uniform'

    output = representation
    SN = True

    for i in range(3):
        output = Linear('{}.linear.{}'.format(name, i), output, DIM//(i+1), SN=SN, initializer=initializer)
        output = LayerNorm1d('{}.Linear.{}'.format(name, i), output, SN=SN)
        output = nonlinearity(output)

    output = Linear('{}.linear.out'.format(name), output, 1, bias=False, SN=SN, initializer='glorot_uniform')

    return output

