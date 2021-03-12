import os, sys, locale, time, random, functools, argparse, pickle as pkl
sys.path.append(os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import scipy
import imageio

import sklearn
from sklearn.cluster import KMeans
import sklearn.metrics

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from utils import *

from networks import feature_learner, predictor
from networks import partition_predictor

######################################## Args ########################################
parser = argparse.ArgumentParser()

parser.add_argument('-dataset', type=str, default='colour', choices=['colour', 'places'])
parser.add_argument('-method', type=str, default='baseline',
        choices=['baseline', 'irmv1', 'rex', 'dro', 'reweight', 'cirmv1', 'crex', 'cdro', 'cmmd', 'pgi'])
parser.add_argument('-sample', type=str, default='balance', choices=['balance', 'normal'])
parser.add_argument('-lr', default=0.1, type=float)
parser.add_argument('-gamma', default=5e-4, type=float)
parser.add_argument('-bs', default=384, type=int)
parser.add_argument('-C0', default=0, type=float)
parser.add_argument('-C1', default=0, type=float)
parser.add_argument('-u', default=10, type=int)
parser.add_argument('-e', default=1, type=int)
parser.add_argument('-r', default=0.8, type=float)
parser.add_argument('-wc', default=0.0, type=float)
parser.add_argument('-v', action='store_true')

args = vars(parser.parse_args())
DATASET = args['dataset']
METHOD = args['method']
BATCH_SIZE = args['bs']
LR = args['lr']
GAMMA = args['gamma']
WC = args['wc']
DRO_C0 = args['C0']
DRO_C1 = args['C1']
RAMP_OVER = args['u']
SAMPLE = args['sample']
EPOCH_AT = args['e']

VALIDATION = args['v']

OUTPUT_DIM = 9
RATIO = args['r']

DROPOUT = None
DIM_CLASS = 64
C, H, W = 3, 64, 64

local_vars = [(k,v) for (k,v) in list(locals().items()) if (k.isupper())]
for var_name, var_value in local_vars:
        print(("\t{}: {}".format(var_name, var_value)))

#######################################################################################
train_handle, id_test_handle, ood_test_handle, sg_test_handle, ano_test_handle = get_handles(OUTPUT_DIM, RATIO, dataset=DATASET, validation=VALIDATION)

train_N, test_N, ano_N = train_handle['y'].shape[0], id_test_handle['y'].shape[0], ano_test_handle['images'].shape[0]

group_gen = in_sequence_train_batch(train_handle, train_N)
train_groups = []
all_labels = []
while True:
    try: _data = next(group_gen)
    except StopIteration: break
    train_groups += [_data[2]]
    all_labels += [_data[1]]
train_groups = np.concatenate(train_groups)
all_labels = np.concatenate(all_labels)

random_inds = sorted(list(np.random.choice(train_N, 2000, replace=False)))
random_sampling = get_train_batch(train_handle, random_inds)
dataset_mean, dataset_std = np.mean(random_sampling[0], (0,2,3), keepdims=True), np.std(random_sampling[0], (0,2,3), keepdims=True)

class_inds = []
for c in range(OUTPUT_DIM):
    class_inds += [np.where(all_labels == c)[0]]

#######################################################################################
EPS = 1e-12

TOTAL_SIZE = train_N

NUM_EPOCHS = 200
C1, C2, C3, C4 = int(0.6*NUM_EPOCHS), int(0.8*NUM_EPOCHS), int(0.9*NUM_EPOCHS), int(0.95*NUM_EPOCHS)
iters_per_epoch = TOTAL_SIZE//BATCH_SIZE

PART_BATCH_SIZE = BATCH_SIZE
PART_ITERS = 100
DIM_PART = DIM_CLASS

PART_START_AT = EPOCH_AT*iters_per_epoch
if METHOD in ['dro', 'cdro']: RAMP_UP = 1
else:                         RAMP_UP = iters_per_epoch*RAMP_OVER
PART_LR = 1e-4

METHODS_FOR_PART = ['pgi', 'irmv1', 'cirmv1', 'dro', 'cdro', 'rex', 'crex', 'reweight', 'cmmd']
#########################################################################################################


#########################################################################################################
def preprocess_images(images):
    images = tf.transpose(images, [0,2,3,1])
    images = tf.pad(images, [[0,0], [H//8,H//8], [W//8,W//8], [0,0]], mode='REFLECT')
    images = tf.map_fn(lambda image: tf.image.random_crop(image, [H,W,C]), images)
    images = tf.map_fn(lambda image: tf.image.random_flip_left_right(image), images)
    images = tf.transpose(images, [0,3,1,2])
    return images

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    images = tf.placeholder(tf.float32, shape=[None, C, H, W], name='images')
    labels = tf.placeholder(tf.int32, shape=[None], name='labels')
    lr = tf.placeholder(tf.float32, shape=None, name='lr')

    T = tf.placeholder(tf.float32, shape=[1,], name='T')
    weight_adapt = tf.placeholder(tf.float32, shape=None, name='weight_adapt')
    dro_weights = tf.placeholder(tf.float32, shape=[2,], name='dro_weights')
    cdro_weights = tf.placeholder(tf.float32, shape=[OUTPUT_DIM,2], name='cdro_weights')
    beta = tf.placeholder(tf.int32, shape=[None,], name='beta')

    is_training = tf.placeholder(tf.bool, shape=None, name='is_training')

    ###################################################################################
    scaled_images = (images - dataset_mean)/dataset_std
    scaled_images = tf.cond(is_training, lambda: preprocess_images(scaled_images), lambda: scaled_images)

    features = feature_learner('theta_f', scaled_images, DIM_CLASS, is_training=is_training, dropout=DROPOUT)
    logits = predictor('theta_p', features, OUTPUT_DIM)
    ###################################################################################


    #############################################################################################################
    xentropy_costs = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(labels, OUTPUT_DIM), logits=logits)
    #############################################################################################################


    ###################################################################################
    sfx = tf.nn.softmax(logits, 1)
    predictive_p = tf.reduce_max(sfx, 1)

    predictions = tf.math.argmax(logits, 1, output_type=tf.dtypes.int32)
    class_partitioned_predictions = tf.dynamic_partition(predictions, labels, OUTPUT_DIM)

    class_accuracies = tf.cast(tf.math.equal(predictions, labels), tf.float32)
    class_accuracy = tf.reduce_mean(class_accuracies)

    feature_train_vars = tf.trainable_variables('theta_f')
    predictor_train_vars = tf.trainable_variables('theta_p')
    train_vars = feature_train_vars + predictor_train_vars

    L2 = tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * GAMMA
    ####################################################################################


    #############################################################################################################
    class_partitioned_betas = tf.dynamic_partition(beta, labels, OUTPUT_DIM)
    class_partitioned_xentropies = tf.dynamic_partition(xentropy_costs, labels, OUTPUT_DIM)
    class_partitioned_features = tf.dynamic_partition(features, labels, OUTPUT_DIM)
    class_partitioned_sfxs = tf.dynamic_partition(sfx, labels, OUTPUT_DIM)
    #############################################################################################################


    ###################################################################################
    alphas, adversary_costs = [], [tf.convert_to_tensor(0.0) for _ in range(OUTPUT_DIM)]
    if METHOD in METHODS_FOR_PART:
        alphas, adversary_costs = [], [tf.convert_to_tensor(0.0) for _ in range(OUTPUT_DIM)]
        dumdum = tf.get_variable('dumdum', shape=(1,OUTPUT_DIM), initializer=tf.constant_initializer(value=1.0), trainable=False)

        for c in range(OUTPUT_DIM):
            alpha_logits = partition_predictor('partition.{}'.format(c), features,  DIM_PART)
            alphas += [0.5*(1.0+tf.nn.tanh(alpha_logits/T))]

            R1 = (1./tf.reduce_sum(alphas[c])) * tf.reduce_sum(alphas[c] * tf.expand_dims(xentropy_costs, 1))
            R2 = (1./tf.reduce_sum(1.0-alphas[c])) * tf.reduce_sum((1.0-alphas[c]) * tf.expand_dims(xentropy_costs, 1))

            h = logits * dumdum
            dummy_costs= tf.losses.softmax_cross_entropy(tf.one_hot(labels, OUTPUT_DIM), h, reduction=tf.losses.Reduction.NONE)

            G1 = tf.reduce_sum(tf.gradients((1./tf.reduce_sum(alphas[c])) * tf.reduce_sum(alphas[c] * tf.expand_dims(dummy_costs, 1)), [dumdum])[0] ** 2)
            G2 = tf.reduce_sum(tf.gradients((1./tf.reduce_sum((1.0-alphas[c]))) * tf.reduce_sum((1.0-alphas[c]) * tf.expand_dims(dummy_costs, 1)), [dumdum])[0] ** 2)

            if DATASET == 'colour': da_cost = R1+R2 + 0.1*(G1+G2)
            elif DATASET == 'places': da_cost = R1+R2 + 0.001*(G1+G2)
            adversary_costs[c] -= 0.5*da_cost
    ###################################################################################


    ####################################################################################
    invariance_penalty = tf.convert_to_tensor(0.0)
    if METHOD == 'pgi':
        for ci in range(OUTPUT_DIM):
            csfxs = class_partitioned_sfxs[ci]
            cbeta = class_partitioned_betas[ci]
            bsfxs = tf.dynamic_partition(csfxs, cbeta, 2)

            def avloss(sfx_0, sfx_1):
                sfx_0 = tf.clip_by_value(tf.reduce_mean(sfx_0, 0), EPS, 1.0-EPS)
                sfx_1 = tf.clip_by_value(tf.reduce_mean(sfx_1, 0), EPS, 1.0-EPS)
                return tf.reduce_mean(sfx_1*tf.log(sfx_1/sfx_0))

            apply_penalty = tf.logical_and(tf.greater(tf.shape(bsfxs[0])[0], 0), tf.greater(tf.shape(bsfxs[1])[0], 0))
            this_is_it = tf.cond(apply_penalty, lambda: avloss(bsfxs[0], bsfxs[1]), lambda: 0.0)
            invariance_penalty += (1./OUTPUT_DIM)*this_is_it

        if SAMPLE == 'balance':
            split_costs = tf.dynamic_partition(xentropy_costs, beta, 2)
            xentropy_cost = tf.cond(tf.logical_and(tf.greater(tf.shape(split_costs[0])[0], 0), tf.greater(tf.shape(split_costs[1])[0], 0)),
                    lambda: 0.5*tf.reduce_mean(split_costs[0]) + 0.5*tf.reduce_mean(split_costs[1]),
                    lambda: tf.reduce_mean(xentropy_costs))
        else:
            xentropy_cost = tf.reduce_mean(xentropy_costs)

    ####################################################################################
    elif METHOD == 'irmv1':
        dummy = tf.get_variable('dummy', shape=(1,OUTPUT_DIM), initializer=tf.constant_initializer(value=1.0), trainable=False)
        env_logits = tf.dynamic_partition(logits, beta, 2)
        env_labels = tf.dynamic_partition(labels, beta, 2)

        def irm_penalty(logits, labels):
            h = logits * dummy
            dummy_cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(labels, OUTPUT_DIM), h, reduction=tf.losses.Reduction.NONE))
            gloss = tf.reduce_sum(tf.gradients(dummy_cost, [dummy])[0] ** 2)
            return gloss

        e0_irm = irm_penalty(env_logits[0], env_labels[0])
        e1_irm = irm_penalty(env_logits[1], env_labels[1])

        invariance_penalty = tf.cond(tf.logical_and(tf.greater(tf.shape(env_logits[0])[0], 0), tf.greater(tf.shape(env_logits[1])[0], 0)),
                lambda: (e0_irm + e1_irm)/2.0,
                lambda: 0.0)

        if SAMPLE == 'balance':
            split_costs = tf.dynamic_partition(xentropy_costs, beta, 2)
            xentropy_cost = tf.cond(tf.logical_and(tf.greater(tf.shape(split_costs[0])[0], 0), tf.greater(tf.shape(split_costs[1])[0], 0)),
                    lambda: 0.5*tf.reduce_mean(split_costs[0]) + 0.5*tf.reduce_mean(split_costs[1]),
                    lambda: tf.reduce_mean(xentropy_costs))
        else:
            xentropy_cost = tf.reduce_mean(xentropy_costs)
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

        if SAMPLE == 'balance':
            split_costs = tf.dynamic_partition(xentropy_costs, beta, 2)
            xentropy_cost = tf.cond(tf.logical_and(tf.greater(tf.shape(split_costs[0])[0], 0), tf.greater(tf.shape(split_costs[1])[0], 0)),
                    lambda: 0.5*tf.reduce_mean(split_costs[0]) + 0.5*tf.reduce_mean(split_costs[1]),
                    lambda: tf.reduce_mean(xentropy_costs))
        else:
            xentropy_cost = tf.reduce_mean(xentropy_costs)
    ####################################################################################


    ####################################################################################
    elif METHOD == 'rex':
        env_losses = tf.dynamic_partition(xentropy_costs, beta, 2)
        loss0, loss1 = tf.reduce_mean(env_losses[0]), tf.reduce_mean(env_losses[1])
        mean_loss = 0.5*(loss0 + loss1)

        invariance_penalty = tf.cond(tf.logical_and(tf.greater(tf.shape(env_losses[0])[0], 0), tf.greater(tf.shape(env_losses[1])[0], 0)),
                lambda: 0.5*((loss0 - mean_loss)**2 + (loss1 - mean_loss)**2), lambda: 0.0)

        if SAMPLE == 'balance':
            split_costs = tf.dynamic_partition(xentropy_costs, beta, 2)
            xentropy_cost = tf.cond(tf.logical_and(tf.greater(tf.shape(split_costs[0])[0], 0), tf.greater(tf.shape(split_costs[1])[0], 0)),
                    lambda: 0.5*tf.reduce_mean(split_costs[0]) + 0.5*tf.reduce_mean(split_costs[1]),
                    lambda: tf.reduce_mean(xentropy_costs))
        else:
            xentropy_cost = tf.reduce_mean(xentropy_costs)
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

        if SAMPLE == 'balance':
            split_costs = tf.dynamic_partition(xentropy_costs, beta, 2)
            xentropy_cost = tf.cond(tf.logical_and(tf.greater(tf.shape(split_costs[0])[0], 0), tf.greater(tf.shape(split_costs[1])[0], 0)),
                    lambda: 0.5*tf.reduce_mean(split_costs[0]) + 0.5*tf.reduce_mean(split_costs[1]),
                    lambda: tf.reduce_mean(xentropy_costs))
        else:
            xentropy_cost = tf.reduce_mean(xentropy_costs)

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

        if SAMPLE == 'balance':
            split_costs = tf.dynamic_partition(xentropy_costs, beta, 2)
            xentropy_cost = tf.cond(tf.logical_and(tf.greater(tf.shape(split_costs[0])[0], 0), tf.greater(tf.shape(split_costs[1])[0], 0)),
                    lambda: 0.5*tf.reduce_mean(split_costs[0]) + 0.5*tf.reduce_mean(split_costs[1]),
                    lambda: tf.reduce_mean(xentropy_costs))
        else:
            xentropy_cost = tf.reduce_mean(xentropy_costs)
    ####################################################################################


    ####################################################################################
    else: xentropy_cost = tf.reduce_mean(xentropy_costs)
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

    if METHOD in ['pgi']:
        optimizer_inv = tf.train.MomentumOptimizer(lr, 0.9)
        gv = optimizer.compute_gradients(xentropy_cost+L2, var_list=train_vars)
        gv_inv = optimizer_inv.compute_gradients(weight_adapt*invariance_penalty, var_list=feature_train_vars)
        with tf.control_dependencies(update_ops):
            class_train_op = tf.group([optimizer.apply_gradients(gv), optimizer_inv.apply_gradients(gv_inv)])
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
    session.run(tf.initialize_all_variables())
    TRAIN_TILL = NUM_EPOCHS*iters_per_epoch

    _lr = LR

    q = np.ones((2,), dtype='float32')
    q = q/np.sum(q)

    cq = np.ones((OUTPUT_DIM,2),dtype='float32')
    cq = cq/np.sum(cq,1,keepdims=True)

    id_accs, ood_accs, sg_accs, av_precs = [], [], [], []
    for iteration in range(TRAIN_TILL):
        ###############################################################################
        if iteration == PART_START_AT:
            if METHOD in METHODS_FOR_PART:
                _partitions = pkl.load(open('./data/predicted_partitions/coco{}_saved_partitions.pkl'.format(DATASET), 'rb'))
                betas = _partitions['betas']
                print('Loaded pre-partitions with partitioning accuracy =', np.mean(betas == train_groups))
        ###############################################################################



        ###############################################################################
        if iteration <= PART_START_AT: wc = 0.0
        else:                          wc = WC*min(1.0, 1.0*(iteration-PART_START_AT+1)/RAMP_UP)

        if iteration == C1*iters_per_epoch:    _lr *= 0.1
        elif iteration == C2*iters_per_epoch:  _lr *= 0.1
        elif iteration == C3*iters_per_epoch:  _lr *= 0.1
        elif iteration == C4*iters_per_epoch:  _lr *= 0.1
        ###############################################################################

        ################################################################################
        inds = sorted(list(np.random.choice(train_N, BATCH_SIZE, replace=False)))
        _data = get_train_batch(train_handle, inds)
        ###############################################################################


        ################################################################################
        if iteration  > PART_START_AT:
            if METHOD in METHODS_FOR_PART: _beta = betas[inds]
            else: _beta = np.ones((BATCH_SIZE,)).astype('int32')
        else: _beta = np.ones((BATCH_SIZE,)).astype('int32')

        ###############################################################################

        ################################################################################
        if METHOD == 'dro' and iteration > PART_START_AT:
            indsb = np.random.choice(np.where(betas == 0)[0], BATCH_SIZE//2, replace=False)
            indsf = np.random.choice(np.where(betas == 1)[0], BATCH_SIZE//2, replace=False)
            inds = sorted(list(np.hstack((indsb, indsf))))
            _data = get_train_batch(train_handle, inds)
            _beta = betas[inds]

            _losses = session.run(xentropy_costs, feed_dict={images: _data[0], labels: _data[1], is_training: True})
            _losses0, _losses1 = _losses[np.where(_beta == 0)[0]], _losses[np.where(_beta == 1)[0]]

            if len(_losses0) > 0 and len(_losses1) > 0:
                _losses0 += DRO_C0/np.sqrt(len(_losses0))
                _losses1 += DRO_C1/np.sqrt(len(_losses1))

                _loss_g = np.array([np.mean(_losses0), np.mean(_losses1)])
                _loss_g = _loss_g/np.sum(_loss_g)

                q = q*np.exp(WC * _loss_g)
                q = q/np.sum(q)
                assert(np.all(q >= 0) and np.all(q <= 1))

            _classifier_cost, _classification_accuracies, _ = session.run([xentropy_cost, class_accuracies, class_train_op],
                    feed_dict={images: _data[0], labels: _data[1], beta: _beta, lr: _lr, dro_weights: q, is_training: True, weight_adapt: 0.0})

        elif METHOD == 'cdro' and iteration > PART_START_AT:
            indsb = np.random.choice(np.where(betas == 0)[0], BATCH_SIZE//2, replace=False)
            indsf = np.random.choice(np.where(betas == 1)[0], BATCH_SIZE//2, replace=False)
            inds = sorted(list(np.hstack((indsb, indsf))))
            _data = get_train_batch(train_handle, inds)
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

            _classifier_cost, _classification_accuracies, _ = session.run([xentropy_cost, class_accuracies, class_train_op],
                    feed_dict={images: _data[0], labels: _data[1], beta: _beta, lr: _lr, cdro_weights: cq, is_training: True, weight_adapt: 0.0})

        else:
            _classifier_cost, _classification_accuracies, _invariance_cost, _ = session.run([xentropy_cost, class_accuracies, invariance_penalty, class_train_op],
                    feed_dict={images: _data[0], labels: _data[1], lr: _lr,
                        beta: _beta,
                        dro_weights: np.ones((2,),dtype='float32'),
                        cdro_weights: np.ones((OUTPUT_DIM,2),dtype='float32'),
                        is_training: True,
                        weight_adapt: wc})
        ###############################################################################


        ################################################################################
        if iteration % 100 == 0 or iteration == (TRAIN_TILL-1):
            id_test_accs, ood_test_accs, sg_test_accs = [], [], []
            normal_pmaxes, abnormal_pmaxes = [], []
            id_test_gen = get_eval_iterator(id_test_handle, test_N, 100, return_groups=True)
            ood_test_gen = get_eval_iterator(ood_test_handle, test_N, 100)
            sg_test_gen = get_eval_iterator(sg_test_handle, test_N, 100)
            ano_test_gen = get_eval_iterator(ano_test_handle, ano_N, 100)

            while True:
                try:
                    id_data = next(id_test_gen)
                    ood_data = next(ood_test_gen)
                    sg_data = next(sg_test_gen)
                except StopIteration: break

                _test_acc_id = session.run(class_accuracies, feed_dict = {images: id_data[0], labels: id_data[1], is_training: False})
                id_test_accs += [_test_acc_id]

                _test_acc_ood, _normal_pmax = session.run([class_accuracy, predictive_p], feed_dict = {images: ood_data[0], labels: ood_data[1], is_training: False})
                normal_pmaxes += [_normal_pmax]
                ood_test_accs += [_test_acc_ood]

                _test_acc_sg = session.run(class_accuracy, feed_dict = {images: sg_data[0], labels: sg_data[1], is_training: False})
                sg_test_accs += [_test_acc_sg]

            ano_data = next(ano_test_gen)   #just one batch of size 100 to simulate rare-ness of anomalies
            _anomaly_pmax = session.run(predictive_p, feed_dict = {images: ano_data[0], is_training: False})
            abnormal_pmaxes += [_anomaly_pmax]

            msp_ins, msp_outs = np.concatenate(normal_pmaxes), np.concatenate(abnormal_pmaxes)
            TRUE_LABELS = np.hstack((np.zeros(len(msp_ins)), np.ones(len(msp_outs))))
            MSP_TESTS = np.hstack((msp_ins, msp_outs))
            msp_av_prec = 100.*sklearn.metrics.average_precision_score(TRUE_LABELS, -MSP_TESTS)

            ##########################################################################

            print(iteration, 'of', TRAIN_TILL, '\t lr =', _lr, 'wc =', wc, end=' ')
            print(': cost = {:.2f}'.format(_classifier_cost), end=' ')
            if METHOD in METHODS_FOR_PART and iteration > PART_START_AT: print('invariance cost = {:.3f}'.format(_invariance_cost), end=' ')
            print(' train = {:.3f}'.format(np.mean(_classification_accuracies)), end=' ')
            print(' ID test = {:.3f}'.format(np.mean(id_test_accs)), end=' ')
            print(' OOD test = {:.3f}'.format(np.mean(ood_test_accs)), end=' ')
            print(' SG test = {:.3f}'.format(np.mean(sg_test_accs)), end=' ')
            print(' Anomaly detection = {:.3f}'.format(msp_av_prec))

            id_accs += [np.mean(id_test_accs)]
            ood_accs += [np.mean(ood_test_accs)]
            sg_accs += [np.mean(sg_test_accs)]
            av_precs += [msp_av_prec]
