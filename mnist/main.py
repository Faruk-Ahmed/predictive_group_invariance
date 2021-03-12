import os, sys, locale, time, random, functools, pickle
sys.path.append(os.getcwd())

from networks import mnist_feature_learner, predictor
from networks import partition_predictor

import numpy as np
import tensorflow as tf

import sklearn.metrics

######################################################################################## 
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-method', type=str, default='baseline', 
        choices=['baseline', 'irmv1', 'rex', 'dro', 'reweight', 'cirmv1', 'crex', 'cdro', 'cmmd', 'pgi'])
parser.add_argument('-c', default=0, type=int)
parser.add_argument('-u', default=1, type=int)
parser.add_argument('-C0', default=0.0, type=float)
parser.add_argument('-C1', default=0.0, type=float)
parser.add_argument('-lr', default=0.1, type=float)
parser.add_argument('-gamma', default=1e-4, type=float)
parser.add_argument('-wc', default=0.0, type=float)
parser.add_argument('-v', action='store_true')

args = vars(parser.parse_args())

METHOD = args['method']
ANOMALY = args['c']
LR = args['lr']
WC = args['wc']
DRO_C0 = args['C0']
DRO_C1 = args['C1']
RAMP_OVER = args['u']
GAMMA = args['gamma']
VALIDATION = args['v']

OUTPUT_DIM = 9
BATCH_SIZE = 512
######################################################################################## 


######################################################################################## 
train_data = pickle.load(open(os.path.join(os.environ['DATA_DIR'], 'cmnist_train_2500.pickle'), 'rb'))

if VALIDATION:
    id_data = pickle.load(open(os.path.join(os.environ['DATA_DIR'], 'cmnist_id_validation_2500.pickle'), 'rb'))
    xn_data = pickle.load(open(os.path.join(os.environ['DATA_DIR'], 'cmnist_xn_validation_2500.pickle'), 'rb'))
    sg_data = pickle.load(open(os.path.join(os.environ['DATA_DIR'], 'cmnist_sg_validation_2500.pickle'), 'rb'))
    TB = 143
else:
    id_data = pickle.load(open(os.path.join(os.environ['DATA_DIR'], 'cmnist_id_test_2500.pickle'), 'rb'))
    xn_data = pickle.load(open(os.path.join(os.environ['DATA_DIR'], 'cmnist_xn_test_2500.pickle'), 'rb'))
    sg_data = pickle.load(open(os.path.join(os.environ['DATA_DIR'], 'cmnist_sg_test_2500.pickle'), 'rb'))
    TB = 451

anomaly_data = pickle.load(open(os.path.join(os.environ['DATA_DIR'], 'cmnist_anomaly_2500.pickle'), 'rb'))
AB = 490

train_images, train_y, train_g = train_data['X'], train_data['y'], train_data['g']
id_test_images, id_test_y = id_data['X'], id_data['y']
xn_test_images, xn_test_y = xn_data['X'], xn_data['y']
sg_test_images, sg_test_y = sg_data['X'], sg_data['y']
anomaly_images, anomaly_fake_labels = anomaly_data['X'], anomaly_data['y']

class_inds = []
for c in range(OUTPUT_DIM):
    class_inds += [np.where(train_y == c)[0]]

train_N = train_images.shape[0]

def in_sequence_train_batch(batch_size):
    assert train_images.shape[0] % batch_size == 0
    for i in range(train_images.shape[0]//batch_size):
        yield (train_images[i*batch_size:(i+1)*batch_size], train_y[i*batch_size:(i+1)*batch_size])

def get_train_batch(batch_size, random_inds=None):
    if random_inds is None: random_inds = np.random.choice(train_images.shape[0], batch_size)
    return (train_images[random_inds], train_y[random_inds])

def get_id_test(bs):
    for i in range(len(id_test_images)//bs):
        yield (id_test_images[i*bs:(i+1)*bs], id_test_y[i*bs:(i+1)*bs])

def get_xn_test(bs):
    for i in range(len(xn_test_images)//bs):
        yield (xn_test_images[i*bs:(i+1)*bs], xn_test_y[i*bs:(i+1)*bs])

def get_sg_test(bs):
    for i in range(len(sg_test_images)//bs):
        yield (sg_test_images[i*bs:(i+1)*bs], sg_test_y[i*bs:(i+1)*bs])

def get_anomalies(bs):
    for i in range(len(anomaly_images)//bs):
        yield (anomaly_images[i*bs:(i+1)*bs],anomaly_fake_labels[i*bs:(i+1)*bs])
######################################################################################## 


######################################################################################## 
EPS = 1e-12 

DIM_CLASS = 64
C, H, W = 3, 28, 28

TOTAL_SIZE = train_images.shape[0]
iters_per_epoch = TOTAL_SIZE//BATCH_SIZE
TRAIN_TILL = 30*iters_per_epoch

PART_LR = 1e-4
PART_BATCH_SIZE = 512
PART_ITERS = 100
DIM_PART = DIM_CLASS
PART_START_AT = iters_per_epoch

METHODS_FOR_PART = ['irmv1', 'dro', 'rex', 'reweight', 'crex', 'cdro', 'cirmv1', 'cmmd', 'pgi']

local_vars = [(k,v) for (k,v) in list(locals().items()) if (k.isupper())]
for var_name, var_value in local_vars:
        print(("\t{}: {}".format(var_name, var_value)))
######################################################################################## 


######################################################################################## 
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    #############################################################################################################
    images = tf.placeholder(tf.uint8, shape=[None, C, H, W], name='images')
    labels = tf.placeholder(tf.int32, shape=[None], name='labels')
    lr = tf.placeholder(tf.float32, shape=None, name='lr')

    T = tf.placeholder(tf.float32, shape=[1,], name='T')
    weight_adapt = tf.placeholder(tf.float32, shape=None, name='weight_adapt')
    dro_weights = tf.placeholder(tf.float32, shape=[2,], name='dro_weights')
    cdro_weights = tf.placeholder(tf.float32, shape=[OUTPUT_DIM,2], name='cdro_weights')
    beta = tf.placeholder(tf.int32, shape=[None,], name='beta')

    is_training = tf.placeholder(tf.bool, shape=None, name='is_training')
    #############################################################################################################

    #############################################################################################################
    scaled_images = 2*(tf.cast(images, tf.float32)/255.0 - 0.5)
    features = mnist_feature_learner('theta_f', scaled_images, DIM_CLASS, is_training=is_training)
    logits = predictor('theta_p', features, OUTPUT_DIM)
    #############################################################################################################


    #############################################################################################################
    xentropy_costs = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(labels, OUTPUT_DIM), logits=logits)
    #############################################################################################################


    ###################################################################################
    sfx = tf.nn.softmax(logits, 1)
    predictive_p = tf.reduce_max(sfx, 1)
    predictions = tf.math.argmax(logits, 1, output_type=tf.dtypes.int32)
    class_accuracy = tf.contrib.metrics.accuracy(labels=labels, predictions=predictions)

    feature_train_vars = tf.trainable_variables('theta_f')
    predictor_train_vars = tf.trainable_variables('theta_p')
    train_vars = feature_train_vars + predictor_train_vars
    L2 = tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * GAMMA
    ####################################################################################


    ###################################################################################
    class_partitioned_betas = tf.dynamic_partition(beta, labels, OUTPUT_DIM)
    class_partitioned_xentropies = tf.dynamic_partition(xentropy_costs, labels, OUTPUT_DIM)
    class_partitioned_features = tf.dynamic_partition(features, labels, OUTPUT_DIM)
    class_partitioned_sfx = tf.dynamic_partition(sfx, labels, OUTPUT_DIM)
    ###################################################################################


    ###################################################################################
    alphas, adversary_costs = [], [tf.convert_to_tensor(0.0) for _ in range(OUTPUT_DIM)]
    if METHOD in METHODS_FOR_PART:
        alphas, adversary_costs = [], [tf.convert_to_tensor(0.0) for _ in range(OUTPUT_DIM)]
        part_dummy = tf.get_variable('part_dummy', shape=(1,OUTPUT_DIM), initializer=tf.constant_initializer(value=1.0), trainable=False)

        for c in range(OUTPUT_DIM):
            alpha_logits = partition_predictor('partition.{}'.format(c), features,  DIM_PART)
            alphas += [0.5*(1.0+tf.nn.tanh(alpha_logits/T))]

            R1 = (1./tf.reduce_sum(alphas[c])) * tf.reduce_sum(alphas[c] * tf.expand_dims(xentropy_costs, 1))
            R2 = (1./tf.reduce_sum(1.0-alphas[c])) * tf.reduce_sum((1.0-alphas[c]) * tf.expand_dims(xentropy_costs, 1))

            h = logits * part_dummy
            dummy_costs= tf.losses.softmax_cross_entropy(tf.one_hot(labels, OUTPUT_DIM), h, reduction=tf.losses.Reduction.NONE)

            G1 = tf.reduce_sum(tf.gradients((1./tf.reduce_sum(alphas[c])) * tf.reduce_sum(alphas[c] * tf.expand_dims(dummy_costs, 1)), [part_dummy])[0] ** 2)
            G2 = tf.reduce_sum(tf.gradients((1./tf.reduce_sum((1.0-alphas[c]))) * tf.reduce_sum((1.0-alphas[c]) * tf.expand_dims(dummy_costs, 1)), [part_dummy])[0] ** 2)

            cpart_cost = R1+R2 + 10*(G1+G2)
            adversary_costs[c] -= 0.5*cpart_cost
    ###################################################################################


    ####################################################################################
    if METHOD == 'pgi':
        invariance_penalty = tf.convert_to_tensor(0.0)

        def pgi_closs(sfx_0, sfx_1):
            sfx_0 = tf.clip_by_value(tf.reduce_mean(sfx_0, 0), EPS, 1.0-EPS)
            sfx_1 = tf.clip_by_value(tf.reduce_mean(sfx_1, 0), EPS, 1.0-EPS)
            return tf.reduce_mean(sfx_1 * tf.log(sfx_1/sfx_0))

        for ci in range(OUTPUT_DIM):
            csfxs = class_partitioned_sfx[ci]
            cbeta = class_partitioned_betas[ci]
            beta_partitioned_sfxs = tf.dynamic_partition(csfxs, cbeta, 2)

            apply_penalty = tf.logical_and(tf.greater(tf.shape(beta_partitioned_sfxs[0])[0], 0), tf.greater(tf.shape(beta_partitioned_sfxs[1])[0], 0))
            invariance_penalty += tf.cond(apply_penalty, 
                    lambda: pgi_closs(beta_partitioned_sfxs[0], beta_partitioned_sfxs[1]),
                    lambda: 0.0)

        split_costs = tf.dynamic_partition(xentropy_costs, beta, 2)
        xentropy_cost = 0.5*tf.cond(tf.greater(tf.shape(split_costs[0])[0], 0), lambda: tf.reduce_mean(split_costs[0]), lambda: 0.0)
        xentropy_cost += 0.5*tf.cond(tf.greater(tf.shape(split_costs[1])[0], 0), lambda: tf.reduce_mean(split_costs[1]), lambda: 0.0)
    ####################################################################################


    ####################################################################################
    elif METHOD == 'irmv1':
        dummy = tf.get_variable('dummy', shape=(1,OUTPUT_DIM), initializer=tf.constant_initializer(value=1.0), trainable=False)
        env_logits = tf.dynamic_partition(logits, beta, 2)
        env_labels = tf.dynamic_partition(labels, beta, 2)

        def irm_penalty(logits, labels):
            h = logits * dummy
            dummy_cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(labels, OUTPUT_DIM), h, reduction=tf.losses.Reduction.NONE))
            return tf.reduce_sum(tf.gradients(dummy_cost, [dummy])[0] ** 2)

        e0_irm = irm_penalty(env_logits[0], env_labels[0])
        e1_irm = irm_penalty(env_logits[1], env_labels[1])

        invariance_penalty = tf.cond(tf.logical_and(tf.greater(tf.shape(env_logits[0])[0], 0), tf.greater(tf.shape(env_logits[1])[0], 0)),
                lambda: (e0_irm + e1_irm)/2.0,
                lambda: 0.0)

        split_costs = tf.dynamic_partition(xentropy_costs, beta, 2)
        xentropy_cost = 0.5*tf.cond(tf.greater(tf.shape(split_costs[0])[0], 0), lambda: tf.reduce_mean(split_costs[0]), lambda: 0.0)
        xentropy_cost += 0.5*tf.cond(tf.greater(tf.shape(split_costs[1])[0], 0), lambda: tf.reduce_mean(split_costs[1]), lambda: 0.0)
    ####################################################################################


    ####################################################################################
    elif METHOD == 'cirmv1':
        invariance_penalty = tf.convert_to_tensor(0.0)
        class_partitioned_logits = tf.dynamic_partition(logits, labels, OUTPUT_DIM)
        class_partitioned_labels = tf.dynamic_partition(labels, labels, OUTPUT_DIM)

        dummy = tf.get_variable('dummy', shape=(1,OUTPUT_DIM), initializer=tf.constant_initializer(value=1.0), trainable=False)
        for ci in range(OUTPUT_DIM):
            cbeta = class_partitioned_betas[ci]
            clogits = class_partitioned_logits[ci]
            clabels = class_partitioned_labels[ci]

            env_logits = tf.dynamic_partition(clogits, cbeta, 2)
            env_labels = tf.dynamic_partition(clabels, cbeta, 2)

            def irm_penalty(_logits, _labels):
                h = _logits * dummy
                dummy_cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(_labels, OUTPUT_DIM), h, reduction=tf.losses.Reduction.NONE))
                return tf.reduce_sum(tf.gradients(dummy_cost, [dummy])[0] ** 2)

            e0_irm = irm_penalty(env_logits[0], env_labels[0])
            e1_irm = irm_penalty(env_logits[1], env_labels[1])

            invariance_penalty += (1./OUTPUT_DIM)*tf.cond(tf.logical_and(tf.greater(tf.shape(env_logits[0])[0], 0), tf.greater(tf.shape(env_logits[1])[0], 0)),
                    lambda: (e0_irm + e1_irm)/2.0,
                    lambda: 0.0)

        split_costs = tf.dynamic_partition(xentropy_costs, beta, 2)
        xentropy_cost = 0.5*tf.cond(tf.greater(tf.shape(split_costs[0])[0], 0), lambda: tf.reduce_mean(split_costs[0]), lambda: 0.0)
        xentropy_cost += 0.5*tf.cond(tf.greater(tf.shape(split_costs[1])[0], 0), lambda: tf.reduce_mean(split_costs[1]), lambda: 0.0)
    ####################################################################################


    ####################################################################################
    elif METHOD == 'rex':
        env_losses = tf.dynamic_partition(xentropy_costs, beta, 2)
        loss0, loss1 = tf.reduce_mean(env_losses[0]), tf.reduce_mean(env_losses[1])
        mean_loss = 0.5*(loss0 + loss1)

        invariance_penalty = tf.cond(tf.logical_and(tf.greater(tf.shape(env_losses[0])[0], 0), tf.greater(tf.shape(env_losses[1])[0], 0)),
                lambda: 0.5*((loss0 - mean_loss)**2 + (loss1 - mean_loss)**2), lambda: 0.0)

        split_costs = tf.dynamic_partition(xentropy_costs, beta, 2)
        xentropy_cost = 0.5*tf.cond(tf.greater(tf.shape(split_costs[0])[0], 0), lambda: tf.reduce_mean(split_costs[0]), lambda: 0.0)
        xentropy_cost += 0.5*tf.cond(tf.greater(tf.shape(split_costs[1])[0], 0), lambda: tf.reduce_mean(split_costs[1]), lambda: 0.0)
    ####################################################################################


    ####################################################################################
    elif METHOD == 'crex':
        invariance_penalty = tf.convert_to_tensor(0.0)
        for ci in range(OUTPUT_DIM):
            closses = class_partitioned_xentropies[ci]
            cbeta = class_partitioned_betas[ci]
            env_losses = tf.dynamic_partition(closses, cbeta, 2)

            loss0, loss1 = tf.reduce_mean(env_losses[0]), tf.reduce_mean(env_losses[1])
            mean_loss = 0.5*(loss0 + loss1)

            invariance_penalty += (1./OUTPUT_DIM)*tf.cond(tf.logical_and(tf.greater(tf.shape(env_losses[0])[0], 0), tf.greater(tf.shape(env_losses[1])[0], 0)),
                    lambda: 0.5*((loss0 - mean_loss)**2 + (loss1 - mean_loss)**2), lambda: 0.0)

        split_costs = tf.dynamic_partition(xentropy_costs, beta, 2)
        xentropy_cost = 0.5*tf.cond(tf.greater(tf.shape(split_costs[0])[0], 0), lambda: tf.reduce_mean(split_costs[0]), lambda: 0.0)
        xentropy_cost += 0.5*tf.cond(tf.greater(tf.shape(split_costs[1])[0], 0), lambda: tf.reduce_mean(split_costs[1]), lambda: 0.0)
    ####################################################################################


    ####################################################################################
    elif METHOD == 'dro':
        invariance_penalty = tf.convert_to_tensor(0.0)
        blosses = tf.dynamic_partition(xentropy_costs, beta, 2)
        apply_penalty = tf.logical_and(tf.greater(tf.shape(blosses[0])[0], 0), tf.greater(tf.shape(blosses[1])[0], 0))
        xentropy_cost = tf.cond(apply_penalty,
                lambda: dro_weights[0]*tf.reduce_mean(blosses[0]) + dro_weights[1]*tf.reduce_mean(blosses[1]),
                lambda: tf.reduce_mean(xentropy_costs))
    ####################################################################################


    ####################################################################################
    elif METHOD == 'cdro':
        invariance_penalty = tf.convert_to_tensor(0.0)
        xentropy_cost = tf.convert_to_tensor(0.0)
        for ci in range(OUTPUT_DIM):
            closses = class_partitioned_xentropies[ci]
            cbeta = class_partitioned_betas[ci]

            blosses = tf.dynamic_partition(closses, cbeta, 2)

            apply_penalty = tf.logical_and(tf.greater(tf.shape(blosses[0])[0], 0), tf.greater(tf.shape(blosses[1])[0], 0))
            xentropy_cost += (1./OUTPUT_DIM)*tf.cond(apply_penalty,
                    lambda: cdro_weights[ci,0]*tf.reduce_mean(blosses[0]) + cdro_weights[ci,1]*tf.reduce_mean(blosses[1]),
                    lambda: tf.reduce_mean(closses))
    ####################################################################################


    ####################################################################################
    elif METHOD == 'reweight':
        invariance_penalty = tf.convert_to_tensor(0.0)
        loss_weights = tf.cast(beta, tf.float32) + (1./(1.0+weight_adapt))*(1-tf.cast(beta, tf.float32))
        loss_weights = loss_weights/tf.reduce_sum(loss_weights)
        xentropy_cost = tf.reduce_sum(loss_weights * xentropy_costs)
    ####################################################################################


    ####################################################################################
    elif METHOD == 'cmmd':
        def kernel(x, y):
            xnorm = tf.reduce_sum(x**2, 1, keepdims=True)
            ynorm = tf.reduce_sum(y**2, 1, keepdims=True)
            distance = xnorm + tf.transpose(ynorm) - 2*tf.matmul(x, tf.transpose(y))

            kernel_llhood = tf.zeros_like(distance)
            bandwidths = [1, 5, 10]
            for s in bandwidths:
                kernel_llhood += (1./len(bandwidths))*tf.exp(-distance/(2*s*s))
            return kernel_llhood
        
        def mmd(x, y):
            K_xx = tf.reduce_mean(kernel(x, x))
            K_yy = tf.reduce_mean(kernel(y, y))
            K_xy = tf.reduce_mean(kernel(x, y))
            return K_xx + K_yy - 2*K_xy

        invariance_penalty = tf.convert_to_tensor(0.0)
        for ci in range(OUTPUT_DIM):
            cfeats = class_partitioned_features[ci]
            cbeta = class_partitioned_betas[ci]
            bfeats = tf.dynamic_partition(cfeats, cbeta, 2)

            apply_penalty = tf.logical_and(tf.greater(tf.shape(bfeats[0])[0], 0), tf.greater(tf.shape(bfeats[1])[0], 0))
            invariance_penalty += (1./OUTPUT_DIM)*tf.cond(apply_penalty, lambda: mmd(bfeats[0], bfeats[1]), lambda: 0.0)

        split_costs = tf.dynamic_partition(xentropy_costs, beta, 2)
        xentropy_cost = 0.5*tf.cond(tf.greater(tf.shape(split_costs[0])[0], 0), lambda: tf.reduce_mean(split_costs[0]), lambda: 0.0)
        xentropy_cost += 0.5*tf.cond(tf.greater(tf.shape(split_costs[1])[0], 0), lambda: tf.reduce_mean(split_costs[1]), lambda: 0.0)
    ####################################################################################


    ####################################################################################
    else:
        invariance_penalty = tf.convert_to_tensor(0.0)
        xentropy_cost = tf.reduce_mean(xentropy_costs)
    ####################################################################################


    ####################################################################################
    classifier_cost  = xentropy_cost + L2 + weight_adapt*invariance_penalty
    ####################################################################################


    ####################################################################################
    if METHOD in ['irmv1', 'rex', 'crex', 'cirmv1']:
        classifier_cost  = tf.cond(tf.greater(weight_adapt, 1.0), lambda: classifier_cost/weight_adapt, lambda: classifier_cost)
    ####################################################################################


    ####################################################################################
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.MomentumOptimizer(lr, 0.9)

    if METHOD == 'pgi':
        feature_grads = tf.gradients(xentropy_cost+L2+weight_adapt*invariance_penalty, feature_train_vars)
        predictor_grads = tf.gradients(xentropy_cost+L2, predictor_train_vars)

        gv = []
        for i, var in enumerate(feature_train_vars):
            gv += [(feature_grads[i], var)]
        for i, var in enumerate(predictor_train_vars):
            gv += [(predictor_grads[i], var)]

        with tf.control_dependencies(feature_grads+predictor_grads+update_ops): 
            class_train_op = optimizer.apply_gradients(gv)
    else:
        gv = optimizer.compute_gradients(classifier_cost, var_list=train_vars)
        with tf.control_dependencies(update_ops):
            class_train_op = optimizer.apply_gradients(gv)
    ###################################################################################


    ###################################################################################
    if METHOD in METHODS_FOR_PART:
        adv_vars, adv_train_ops = [], []
        for c in range(OUTPUT_DIM):
            adversary_optimizer = tf.train.AdamOptimizer(PART_LR, name='advopt{}'.format(c))
            adv_vars += [tf.trainable_variables('partition.{}'.format(c))]

            adversary_gv = adversary_optimizer.compute_gradients(adversary_costs[c], var_list=adv_vars[c])
            adv_train_ops += [adversary_optimizer.apply_gradients(adversary_gv)]
    ####################################################################################


    ####################################################################################
    def signal(_data):
        _msp_signal = session.run(predictive_p, feed_dict={images: _data[0], labels: _data[1], is_training: False})
        return -_msp_signal

    def compute_av_prec():
        msp_ins, msp_outs, out_accs = [], [], []
        in_test_gen, out_test_gen = get_xn_test(TB), get_anomalies(AB)

        while True:
            try: _data = next(in_test_gen)
            except StopIteration: break
            msp_scores = signal(_data)
            msp_ins += [msp_scores]

        while True:
            try: _data = next(out_test_gen)
            except StopIteration: break

            msp_scores = signal(_data)
            msp_outs += [msp_scores]

        msp_ins, msp_outs = np.concatenate(msp_ins), np.concatenate(msp_outs)

        TRUE_LABELS = np.hstack((np.zeros(len(msp_ins)), np.ones(len(msp_outs))))
        MSP_TESTS = np.hstack((msp_ins, msp_outs))

        return 100.*sklearn.metrics.average_precision_score(TRUE_LABELS, MSP_TESTS)
    ####################################################################################


    ####################################################################################
    session.run(tf.initialize_all_variables())
    ra_cost, ra_acc = 0.0, 0.0

    _lr = LR
    if METHOD in ['dro', 'cdro']: RAMP_UP = 1
    else: RAMP_UP = iters_per_epoch*RAMP_OVER

    for iteration in range(TRAIN_TILL):

        if iteration <= PART_START_AT: wc = 0.0
        else:                          wc = WC*min(1.0, 1.0*(iteration-PART_START_AT+1)/RAMP_UP)

        if iteration == 9*iters_per_epoch:    _lr *= 0.1
        elif iteration == 18*iters_per_epoch: _lr *= 0.1
        elif iteration == 24*iters_per_epoch: _lr *= 0.1

        ###############################################################################
        if iteration == PART_START_AT:
            if METHOD in METHODS_FOR_PART:
                for c in range(OUTPUT_DIM):
                    print(' -- Splitting', c)
                    session.run(tf.variables_initializer(adv_vars[c]))
                    for adv_iter in range(PART_ITERS):
                        inds = sorted(list(np.random.choice(class_inds[c], PART_BATCH_SIZE, replace=False)))
                        _l_phi, _ = session.run([adversary_costs[c], adv_train_ops[c]], 
                                feed_dict={images: train_images[inds,...], labels: train_y[inds,...], is_training: False, T: [10.0]})

                beta_gen = in_sequence_train_batch(1)
                betas, _losses, _classes = [], [], []
                while True:
                    try: _data = next(beta_gen)
                    except StopIteration: break
                    _loss, _beta = session.run([xentropy_costs, alphas[_data[1][0]]],
                            feed_dict={images: _data[0], labels: _data[1], is_training: False, T: [10.0]})

                    betas += [np.round(_beta)]
                    _losses += [_loss]
                    _classes += [_data[1]]

                betas = np.concatenate(betas)[:,0]
                _losses, _classes = np.concatenate(_losses), np.concatenate(_classes)

                for c in range(OUTPUT_DIM):
                    cinds = np.where(_classes == c)[0]
                    _closses = _losses[cinds]
                    _cbetas = betas[cinds]
                    
                    if np.sum(_cbetas) > 0 and np.sum(_cbetas) < len(_cbetas):
                        _c0, _c1 = np.mean(_closses[np.where(_cbetas == 0)]), np.mean(_closses[np.where(_cbetas == 1)])
                        if _c1 < _c0: 
                            betas[cinds] = 1 - betas[cinds]

                # Or you can load betas = pkl.load(open('./data/predicted_partitions/saved_partitions.pkl', 'rb'))['betas']

                Nu = np.sum(betas)
                Nb = len(betas)-Nu

                Nu, Nb = np.where(betas == 1)[0].shape[0], np.where(betas == 0)[0].shape[0]

                q = np.ones((len(np.unique(betas)),),dtype='float32')
                q = q/np.sum(q)

                cq = np.ones((OUTPUT_DIM,2),dtype='float32')
                cq = cq/np.sum(cq,1,keepdims=True)
        ################################################################################

        inds = sorted(list(np.random.choice(train_N, BATCH_SIZE, replace=False)))
        _data = get_train_batch(BATCH_SIZE, inds)

        if iteration  >= PART_START_AT:
            if METHOD in METHODS_FOR_PART:  _beta = betas[inds]
            elif METHOD == 'baseline':      _beta = np.ones((BATCH_SIZE,)).astype('int32')
        else: _beta = np.ones((BATCH_SIZE,)).astype('int32')

        if METHOD == 'dro' and iteration >= PART_START_AT:
            # DRO recommends some sort of balanced sampling; this seems to work best overall
            indsb = np.random.choice(np.where(betas == 0)[0], BATCH_SIZE//2, replace=False)
            indsf = np.random.choice(np.where(betas == 1)[0], BATCH_SIZE//2, replace=False)
            inds = sorted(list(np.hstack((indsb, indsf))))
            _data = get_train_batch(BATCH_SIZE, inds)
            _beta = betas[inds]

            _losses = session.run(xentropy_costs, feed_dict={images: _data[0], labels: _data[1], is_training: True})
            _losses0, _losses1 = _losses[np.where(_beta == 0)[0]], _losses[np.where(_beta == 1)[0]]

            _losses0 += DRO_C0/np.sqrt(len(_losses0))
            _losses1 += DRO_C1/np.sqrt(len(_losses1))

            _loss_g = np.array([np.mean(_losses0), np.mean(_losses1)])

            # Normalising here seemed to help in early experiments overall
            _loss_g = _loss_g/np.sum(_loss_g)  

            q = q*np.exp(WC * _loss_g)
            q = q/np.sum(q)

            _classifier_cost, _classification_accuracy, _ = session.run([xentropy_cost, class_accuracy, class_train_op],
                    feed_dict={images: _data[0], labels: _data[1], beta: _beta, lr: _lr, dro_weights: q, is_training: True, weight_adapt: 0.0})

        elif METHOD == 'cdro' and iteration >= PART_START_AT:
            indsb = np.random.choice(np.where(betas == 0)[0], BATCH_SIZE//2, replace=False)
            indsf = np.random.choice(np.where(betas == 1)[0], BATCH_SIZE//2, replace=False)
            inds = sorted(list(np.hstack((indsb, indsf))))
            _data = get_train_batch(BATCH_SIZE, inds)
            _beta = betas[inds]

            _losses = session.run(xentropy_costs, feed_dict={images: _data[0], labels: _data[1], is_training: True})
            for ci in range(OUTPUT_DIM):
                cinds0 = np.where((_data[1] == ci) & (_beta == 0))[0]
                cinds1 = np.where((_data[1] == ci) & (_beta == 1))[0]

                _losses0, _losses1 = _losses[cinds0], _losses[cinds1]

                if len(_losses0) > 0 and len(_losses1) > 0:
                    _losses0 += DRO_C0/np.sqrt(len(_losses0))
                    _losses1 += DRO_C1/np.sqrt(len(_losses1))

                    _loss_g = np.array([np.mean(_losses0), np.mean(_losses1)])
                    _loss_g = _loss_g/np.sum(_loss_g)

                    cq[ci,:] = cq[ci,:]*np.exp(WC * _loss_g)
                    cq[ci,:] = cq[ci,:]/np.sum(cq[ci,:])

            _classifier_cost, _classification_accuracy, _ = session.run([xentropy_cost, class_accuracy, class_train_op],
                    feed_dict={images: _data[0], labels: _data[1], beta: _beta, lr: _lr, cdro_weights: cq, is_training: True, weight_adapt: 0.0})

        else:
            _classifier_cost, _classification_accuracy, _invariance_cost, _ = session.run([xentropy_cost, class_accuracy, invariance_penalty, class_train_op],
                    feed_dict={images: _data[0], labels: _data[1], lr: _lr, 
                        beta: _beta, 
                        dro_weights: np.ones((2,),dtype='float32'), 
                        cdro_weights: np.ones((OUTPUT_DIM,2),dtype='float32'), 
                        is_training: True, 
                        weight_adapt: wc})

        ra_cost += (_classifier_cost - ra_cost)/(iteration+1)
        ra_acc += (_classification_accuracy - ra_acc)/(iteration+1)

        ################################################################################
        if iteration % 100 == 0 or iteration == (TRAIN_TILL-1):
            _id_test_acc, _xn_test_acc, _sg_test_acc, _test_iter = 0, 0, 0, 0
            id_test_gen = get_id_test(TB)
            xn_test_gen = get_xn_test(TB)
            sg_test_gen = get_sg_test(TB)

            while True:
                try: 
                    _id_data = next(id_test_gen)
                    _xn_data = next(xn_test_gen)
                    _sg_data = next(sg_test_gen)
                except StopIteration: break

                _id_acc = session.run(class_accuracy, feed_dict = {images: _id_data[0], labels: _id_data[1], is_training: False})
                _xn_acc = session.run(class_accuracy, feed_dict = {images: _xn_data[0], labels: _xn_data[1], is_training: False})
                _sg_acc = session.run(class_accuracy, feed_dict = {images: _sg_data[0], labels: _sg_data[1], is_training: False})

                _id_test_acc += (_id_acc - _id_test_acc)/(_test_iter+1)
                _xn_test_acc += (_xn_acc - _xn_test_acc)/(_test_iter+1)
                _sg_test_acc += (_sg_acc - _sg_test_acc)/(_test_iter+1)
                _test_iter += 1

            av_prec = compute_av_prec()
            ##########################################################################

            print(iteration, 'of', TRAIN_TILL, '\t lr =', _lr, 'wc =', wc, end=' ')
            print(': cost = {:.2f}'.format(ra_cost), end=' ') 
            if METHOD in METHODS_FOR_PART: print('invariance cost = {:.4f}'.format(_invariance_cost), end=' ')
            print('train = {:.3f}'.format(ra_acc), 'id_test = {:.3f}'.format(_id_test_acc), 'xn = {:.3f}'.format(_xn_test_acc), 'sg = {:.3f}'.format(_sg_test_acc), end=' ')
            print('av pr =', av_prec)
