import os, sys
sys.path.append(os.getcwd())

from layers import Linear
from layers import LinearSN
from layers import Conv2D
from layers import Normalize
from layers import LayerNorm1d
from layers import ResidualLayer

import numpy as np
import tensorflow as tf

from functools import partial


def feature_learner(name, images, DIM, dropout=None, is_training=None):
    output = images

    output = Conv2D('{}.conv.Init'.format(name), output, DIM, initializer='glorot_uniform')

    output = ResidualLayer('{}.conv.1.1'.format(name), output, DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('{}.conv.1.2'.format(name), output, DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('{}.conv.1.3'.format(name), output, DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('{}.conv.1.4'.format(name), output, 2*DIM, is_training=is_training, dropout=dropout)

    output = ResidualLayer('{}.conv.2.1'.format(name), output, 2*DIM, stride=2, is_training=is_training, dropout=dropout)
    output = ResidualLayer('{}.conv.2.2'.format(name), output, 2*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('{}.conv.2.3'.format(name), output, 2*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('{}.conv.2.4'.format(name), output, 4*DIM, is_training=is_training, dropout=dropout)

    output = ResidualLayer('{}.conv.3.1'.format(name), output, 4*DIM, stride=2, is_training=is_training, dropout=dropout)
    output = ResidualLayer('{}.conv.3.2'.format(name), output, 4*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('{}.conv.3.3'.format(name), output, 4*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('{}.conv.3.4'.format(name), output, 4*DIM, is_training=is_training, dropout=dropout)

    output = ResidualLayer('{}.conv.4.1'.format(name), output, 4*DIM, stride=2, is_training=is_training, dropout=dropout)
    output = ResidualLayer('{}.conv.4.2'.format(name), output, 4*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('{}.conv.4.3'.format(name), output, 4*DIM, is_training=is_training, dropout=dropout)
    output = ResidualLayer('{}.conv.4.4'.format(name), output, 4*DIM, is_training=is_training, dropout=dropout)

    output = Normalize('{}.convout.3.NORM'.format(name), output, bn_is_training=is_training)
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

