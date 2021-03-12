import numpy as np
import tensorflow as tf

_INITIALIZERS = {'glorot_uniform': tf.glorot_uniform_initializer(),
                 'glorot_normal': tf.glorot_normal_initializer(),
                 'orthogonal': tf.orthogonal_initializer(),
                 'lecun_normal': tf.initializers.lecun_normal(),
                 'lecum_uniform': tf.initializers.lecun_uniform(),
                 'he_normal': tf.initializers.he_normal(),
                 'he_uniform': tf.initializers.he_uniform(),
                 'constant_one': tf.constant_initializer(value=1.0),
                 'constant_zero': tf.constant_initializer(value=0.0),
                 }

_BIAS_INITIALIZER = tf.constant_initializer(value=0.01)
_DEFAULT_INIT = 'glorot_normal'
_DEFAULT_NORM = 'BN'

def Linear(name, x, output_dim, bias=True, SN=False, initializer=_DEFAULT_INIT):
    if SN: return LinearSN(name, x, output_dim, bias=bias, initializer=initializer, power_iters=1)

    input_dim = x.get_shape()[1]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('Filters',
            shape=[input_dim, output_dim], 
            initializer=_INITIALIZERS[initializer],
            trainable=True)

    if bias:
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            biases = tf.get_variable('Biases',
                    shape=[output_dim],
                    initializer=_BIAS_INITIALIZER,
                    trainable=True)
        out = tf.add(tf.matmul(x, weights, name='Matmul'), biases, name='Output')
    else:
        out = tf.matmul(x, weights, name='Matmul')

    return out

def norm(x, axes=[1,2,3]):
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axes, keepdims=True))

def LinearSN(name, x, output_dim, bias=True, initializer=_DEFAULT_INIT, power_iters=1):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        def spectral_normalization(W):
            W_shape = W.shape.as_list()
            u = tf.get_variable('u', 
                    [1, W_shape[0]], 
                    initializer=tf.random_normal_initializer(), 
                    trainable=False)

            v_ = tf.matmul(u, W)
            v_hat = v_/norm(v_, [1]) 

            u_ = tf.matmul(v_hat, tf.transpose(W))
            u_hat = u_/norm(u_, [1]) 

            u_hat = tf.stop_gradient(u_hat)
            v_hat = tf.stop_gradient(v_hat)

            sigma = tf.reduce_sum(u_hat * tf.matmul(v_hat, tf.transpose(W)), axis=1)

            with tf.control_dependencies([u.assign(u_hat)]):
                W_norm = W/sigma

            return W_norm

        shape = x.get_shape().as_list()
        input_dim = shape[1]

        weights = tf.get_variable('Filters',
                shape=[input_dim, output_dim], 
                initializer=_INITIALIZERS[initializer],
                trainable=True)

        biases = tf.get_variable('Biases',
                shape=[output_dim],
                initializer=_BIAS_INITIALIZER,
                trainable=True)

        out = tf.matmul(x, spectral_normalization(weights), name='Matmul')
        if bias: out = tf.add(out, biases, name='Output')

    return out

def Conv2D(name, x, output_dim, kernel_size=3, stride=1, padding='SAME', bias=True, initializer=_DEFAULT_INIT):
    shape = x.get_shape().as_list()
    input_dim = shape[1]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weights = tf.get_variable('Filters',
                shape=[kernel_size, kernel_size, input_dim, output_dim], 
                initializer=_INITIALIZERS[initializer],
                trainable=True)
    if bias:
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            biases = tf.get_variable('Biases',
                    shape=[output_dim],
                    initializer=_BIAS_INITIALIZER,
                    trainable=True)

    strides = [1,1,stride, stride]
    out = tf.nn.conv2d(x, weights, strides, padding, name='Conv2d', data_format='NCHW')
    if bias: out = tf.nn.bias_add(out, biases, data_format='NCHW')

    return out


def LayerNorm1d(name, x, SN=False):
    x = tf.convert_to_tensor(x)
    x_shape = x.get_shape()
    num_channels = x.get_shape().as_list()[1]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if not SN: gamma = tf.get_variable('LN.Gamma', shape=[num_channels], initializer=tf.ones_initializer(), trainable=True)
        beta = tf.get_variable('LN.Beta', shape=[num_channels], initializer=tf.zeros_initializer(), trainable=True)

    mean, variance = tf.nn.moments(x, [1], keep_dims=True)
    if not SN: output = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=gamma, variance_epsilon=1e-5)
    else:      output = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=tf.ones([num_channels]), variance_epsilon=1e-5)

    return output

def Normalize(name, x, method=_DEFAULT_NORM, bn_is_training=True):
    if method == 'BN':
        num_channels = x.get_shape().as_list()[1]

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable('Gamma', 
                    shape=[num_channels], 
                    initializer=tf.ones_initializer(), 
                    trainable=True)

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            biases = tf.get_variable('Beta', 
                    shape=[num_channels], 
                    initializer=tf.zeros_initializer(),
                    trainable=True)

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            moving_mean = tf.get_variable('FBN.Moving_mean', 
                    shape=[num_channels], 
                    trainable=False, 
                    initializer=tf.zeros_initializer())
            moving_variance = tf.get_variable('FBN.Moving_variance', 
                    shape=[num_channels], 
                    trainable=False, 
                    initializer=tf.ones_initializer())

        def _bn_training():
            training_bn_output, mean, variance = tf.nn.fused_batch_norm(x, 
                     scale=weights, 
                     offset=biases, 
                     epsilon=1e-5, 
                     data_format='NCHW')
            momentum = 0.9
            update_moving_mean = (1.-momentum)*moving_mean + momentum*mean
            update_moving_variance = (1.-momentum)*moving_variance + momentum*variance
            
            update_ops = [moving_mean.assign(update_moving_mean), moving_variance.assign(update_moving_variance)]
            with tf.control_dependencies(update_ops):
                return tf.identity(training_bn_output)
                 
        def _bn_inference():
            inference_bn_output, _, _ = tf.nn.fused_batch_norm(x,
                    mean=moving_mean, 
                    variance=moving_variance,
                    scale=weights, 
                    offset=biases,
                    is_training=False,
                    epsilon=1e-5, 
                    data_format='NCHW')
            return inference_bn_output

        output = tf.cond(bn_is_training, lambda: _bn_training(), lambda: _bn_inference())
        return output

    elif method == 'LN':
        x = tf.convert_to_tensor(x)
        x_shape = x.get_shape()
        num_channels = x.get_shape().as_list()[1]

        with tf.variable_scope(name+'.LN', reuse=tf.AUTO_REUSE):
            weights = tf.get_variable('Gamma', shape=[1,num_channels,1,1], initializer=tf.ones_initializer(), trainable=True)

        with tf.variable_scope(name+'.LN', reuse=tf.AUTO_REUSE):
            biases = tf.get_variable('LN.Beta', shape=[1,num_channels,1,1], initializer=tf.zeros_initializer(), trainable=True)

        mean, variance = tf.nn.moments(x, [1,2,3], keep_dims=True)
        output = tf.nn.batch_normalization(x, mean, variance, biases, weights, 1e-5)
        return output

    elif method == 'GN':
        _shape = tf.shape(x)
        N, C, H, W = _shape[0], _shape[1], _shape[2], _shape[3]
        G = tf.math.minimum(8, C)
        _C = x.get_shape()[1]

        output = tf.transpose(x, [0,2,3,1])
        output = tf.reshape(output, [N,H,W,G,C//G])
        mean, var = tf.nn.moments(output, [1,2,4], keep_dims=True)
        output = (output - mean) / tf.sqrt(var + 1e-5)

        with tf.variable_scope(name+'.GN', reuse=tf.AUTO_REUSE):
            weights = tf.get_variable('Gamma', [1,1,1,_C], initializer=tf.ones_initializer(), trainable=True)

        with tf.variable_scope(name+'.GN', reuse=tf.AUTO_REUSE):
            biases = tf.get_variable('Beta', [1,1,1,_C], initializer=tf.zeros_initializer(), trainable=True)

        return tf.transpose(tf.reshape(output, [N,H,W,C]) * weights + biases, [0,3,1,2])


def ResidualLayer(name, x, output_dim, kernel_size=3, stride=1, norm=_DEFAULT_NORM, is_training=None, dropout=None, weights=None, biases=None):
    shape = x.get_shape().as_list()
    input_dim = shape[1]

    output = Normalize(name+'.NORM.L1', x, norm, is_training)
    output = tf.nn.relu(output)

    if input_dim == output_dim and stride == 1: shortcut = x 
    else: shortcut = Conv2D(name+'.shortcut', x, output_dim, 1, stride, bias=False, initializer='glorot_uniform')

    output = Conv2D(name+'.CONV.L1', output, output_dim, kernel_size, stride, bias=False)
    output = Normalize(name+'.NORM.L2', output, norm, is_training)
    output = tf.nn.relu(output)

    if dropout is not None: output = tf.layers.dropout(output, rate=dropout, training=is_training)

    output = Conv2D(name+'.CONV.L2', output, output_dim, kernel_size, bias=False)
    return shortcut + output

def ResidualLayer1d(name, x, output_dim, initializer='glorot_uniform'):
    shape = x.get_shape().as_list()
    input_dim = shape[1]

    output = LayerNorm1d(name+'.NORM.L1', x)
    output = tf.nn.relu(output)

    if input_dim == output_dim: shortcut = x
    else: shortcut = LinearSN(name+'.shortcut', x, output_dim, initializer='glorot_uniform')

    output = LinearSN(name+'.Linear.L1', output, output_dim, bias=False, initializer=initializer)
    output = LayerNorm1d(name+'.NORM.L2', output)
    output = tf.nn.relu(output)

    output = LinearSN(name+'.Linear.L2', output, output_dim, bias=False, initializer=initializer)

    return shortcut + output

