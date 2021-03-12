import os, sys, locale
sys.path.append(os.getcwd())
import numpy as np
import time
import random
import functools
import gzip, pickle as pickle


######################################### Args ############################################
BIAS_RATIO = 0.8
ANOMALY = 0

##################### Bit of customizing the dataloaders for this #########################
def shift_labels(y):
    sub = np.where(y > ANOMALY, 1, 0)
    return y - sub

filepath = os.path.join(os.environ['DATA_DIR'], 'mnist.pkl.gz')

with gzip.open(filepath, 'rb') as f:
    train_data, dev_data, test_data = pickle.load(f, encoding='latin1')

train_x, train_y = np.float32(train_data[0] > 0.5), train_data[1]
dev_x, dev_y = np.float32(dev_data[0] > 0.5), dev_data[1]
test_x, test_y = np.float32(test_data[0] > 0.5), test_data[1]

###########################################################################################
train_normal_inds = np.in1d(train_y, np.setdiff1d(list(range(10)), ANOMALY))
normal_train_x, normal_train_y = train_x[train_normal_inds], shift_labels(train_y[train_normal_inds])

dev_normal_inds = np.in1d(dev_y, np.setdiff1d(list(range(10)), ANOMALY))
normal_dev_x, normal_dev_y = dev_x[dev_normal_inds], shift_labels(dev_y[dev_normal_inds])

test_normal_inds = np.in1d(test_y, np.setdiff1d(list(range(10)), ANOMALY))
normal_test_x, normal_test_y = test_x[test_normal_inds], shift_labels(test_y[test_normal_inds])

test_anomaly_inds = np.in1d(test_y, ANOMALY)
anomaly_test_x = test_x[test_anomaly_inds]

###########################################################################################
biasing_colours = [[0,100,0],
                   [188, 143, 143],
                   [255, 0, 0],
                   [255, 215, 0],
                   [0, 255, 0],
                   [65, 105, 225],
                   [0, 225, 225],
                   [0, 0, 255],
                   [255, 20, 147]]
biasing_colours = np.array(biasing_colours)

# Make images 3-channel (for colorizing):
normal_train_images = np.tile(np.reshape(normal_train_x, [-1, 1, 28, 28]), [1,3,1,1])
normal_dev_images = np.tile(np.reshape(normal_dev_x, [-1, 1, 28, 28]), [1,3,1,1])
normal_test_images = np.tile(np.reshape(normal_test_x, [-1, 1, 28, 28]), [1,3,1,1])
anomaly_test_images = np.tile(np.reshape(anomaly_test_x, [-1, 1, 28, 28]), [1,3,1,1])


###########################################################################################
_D = 2500
def random_different_enough_colour():
    while True:
        x = np.random.choice(255, size=3)
        if np.min(np.sum((x - biasing_colours)**2, 1)) > _D: # and np.sum(x) > 50:
            break
    return list(x)
unbiasing_colours = np.array([random_different_enough_colour() for _ in range(10)])

def test_colours():
    while True:
        x = np.random.choice(255, size=3)
        if np.min(np.sum((x - biasing_colours)**2, 1)) > _D and np.min(np.sum((x - unbiasing_colours)**2, 1)) > _D: # and np.sum(x) > 50:
            break
    return x
utest_colours = np.array([test_colours() for _ in range(10)])

def validation_colours():
    while True:
        x = np.random.choice(255, size=3)
        if np.min(np.sum((x - biasing_colours)**2, 1)) > _D and np.min(np.sum((x - unbiasing_colours)**2, 1)) > _D and np.min(np.sum((x - utest_colours)**2, 1)) > _D: # and np.sum(x) > 50:
            break
    return x
uvalidation_colours = np.array([validation_colours() for _ in range(10)])

###########################################################################################
def get_train(bs, bias_ratio=BIAS_RATIO):
    assert normal_train_images.shape[0] % bs == 0
    for i in range(len(normal_train_images) // bs):
        _im = np.copy(normal_train_images[i*bs:(i+1)*bs])
        _l = normal_train_y[i*bs:(i+1)*bs]

        for m in range(_im.shape[0]):
            if np.random.binomial(1,BIAS_RATIO) == 1:
                _im[m, ...] = np.multiply(_im[m, ...], biasing_colours[_l[m]][:,None,None])
                _g = np.array([0])
            else:
                random_colour = unbiasing_colours[np.random.choice(unbiasing_colours.shape[0])][:,None,None]
                _im[m, ...] = np.multiply(_im[m,...], random_colour)
                _g = np.array([1])

        yield (_im, _l, _g)

def get_test(bs, bias_ratio=BIAS_RATIO):
    assert normal_test_images.shape[0] % bs == 0
    for i in range(len(normal_test_images) // bs):
        _im = np.copy(normal_test_images[i*bs:(i+1)*bs])
        _l = normal_test_y[i*bs:(i+1)*bs]

        for m in range(_im.shape[0]):
            if np.random.binomial(1,BIAS_RATIO) == 1:
                _im[m, ...] = np.multiply(_im[m, ...], biasing_colours[_l[m]][:,None,None])
            else:
                random_colour = unbiasing_colours[np.random.choice(unbiasing_colours.shape[0])][:,None,None]
                _im[m, ...] = np.multiply(_im[m,...], random_colour)
        yield (_im, _l)


def get_id_valid(bs, bias_ratio=BIAS_RATIO):
    assert normal_dev_images.shape[0] % bs == 0
    for i in range(len(normal_dev_images) // bs):
        _im = np.copy(normal_dev_images[i*bs:(i+1)*bs])
        _l = normal_dev_y[i*bs:(i+1)*bs]

        for m in range(_im.shape[0]):
            if np.random.binomial(1,BIAS_RATIO) == 1:
                _im[m, ...] = np.multiply(_im[m, ...], biasing_colours[_l[m]][:,None,None])
            else:
                random_colour = unbiasing_colours[np.random.choice(unbiasing_colours.shape[0])][:,None,None]
                _im[m, ...] = np.multiply(_im[m,...], random_colour)
        yield (_im, _l)

def get_xn_valid(bs, bias_ratio=BIAS_RATIO):
    assert normal_dev_images.shape[0] % bs == 0
    for i in range(len(normal_dev_images) // bs):
        _im = np.copy(normal_dev_images[i*bs:(i+1)*bs])
        _l = normal_dev_y[i*bs:(i+1)*bs]

        for m in range(_im.shape[0]):
            random_colour = uvalidation_colours[np.random.choice(uvalidation_colours.shape[0])][:,None,None]
            _im[m, ...] = np.multiply(_im[m,...], random_colour)
        yield (_im, _l)

def get_sg_valid(bs, bias_ratio=BIAS_RATIO):
    assert normal_dev_images.shape[0] % bs == 0
    for i in range(len(normal_dev_images) // bs):
        _im = np.copy(normal_dev_images[i*bs:(i+1)*bs])
        _l = normal_dev_y[i*bs:(i+1)*bs]

        for m in range(_im.shape[0]):
            idx = np.random.choice(list(set(range(9))-set([_l[m]])))
            random_colour = biasing_colours[idx][:, None, None]
            _im[m, ...] = (_im[m, ...] * random_colour)
        yield (_im, _l)

def get_xn(batch_size):
    assert normal_test_images.shape[0] % batch_size == 0
    for i in range(len(normal_test_images) // batch_size):
        _im = np.copy(normal_test_images[i*batch_size:(i+1)*batch_size])
        _l = normal_test_y[i*batch_size:(i+1)*batch_size]

        for m in range(_im.shape[0]):
            random_colour = utest_colours[np.random.choice(utest_colours.shape[0])][:,None,None]
            _im[m, ...] = np.multiply(_im[m,...], random_colour)

        yield (_im, _l)

def get_sg(bs=1):
    for i in range(len(normal_test_images)):
        _im = np.copy(normal_test_images[i:(i+1)])
        _l = normal_test_y[i:(i+1)]

        idx = np.random.choice(list(set(range(9))-set(_l)))
        random_colour = biasing_colours[idx][None,:,None,None]
        _im = np.multiply(_im, random_colour)

        yield (_im, _l)

def get_anomalies(bs=1):
    for i in range(len(anomaly_test_images)):
        _im = np.copy(anomaly_test_images[i:(i+1)])
        _l = np.random.choice(9)
        random_colours = biasing_colours[_l][None,:,None,None]
        _im = (_im * random_colours)

        yield (_im,_l)


##### Make data dumps:
normal_training_set_to_dump = {'X': [], 'y': [], 'g': []}
id_dev_set_to_dump = {'X': [], 'y': []}
xn_dev_set_to_dump = {'X': [], 'y': []}
sg_dev_set_to_dump = {'X': [], 'y': []}
id_test_set_to_dump = {'X': [], 'y': []}
xn_test_set_to_dump = {'X': [], 'y': []}
sg_test_set_to_dump = {'X': [], 'y': []}
anomaly_test_set_to_dump = {'X': [], 'y': []}

train_gen = get_train(1)

id_dev_gen = get_id_valid(1)
xn_dev_gen = get_xn_valid(1)
sg_dev_gen = get_sg_valid(1)

id_test_gen = get_test(1)
xn_test_gen = get_xn(1)
sg_test_gen = get_sg(1)
anomaly_gen = get_anomalies(1)

print('Training data')
while True:
    try: _data = next(train_gen)
    except StopIteration: break
    normal_training_set_to_dump['X'] += [_data[0]]
    normal_training_set_to_dump['y'] += [_data[1]]
    normal_training_set_to_dump['g'] += [_data[2]]
normal_training_set_to_dump['X'] = np.concatenate(normal_training_set_to_dump['X'])
normal_training_set_to_dump['y'] = np.concatenate(normal_training_set_to_dump['y'])
normal_training_set_to_dump['g'] = np.concatenate(normal_training_set_to_dump['g'])

print('ID Dev data')
while True:
    try: _data = next(id_dev_gen)
    except StopIteration: break
    id_dev_set_to_dump['X'] += [_data[0]]
    id_dev_set_to_dump['y'] += [_data[1]]
id_dev_set_to_dump['X'] = np.concatenate(id_dev_set_to_dump['X'])
id_dev_set_to_dump['y'] = np.concatenate(id_dev_set_to_dump['y'])

print('Xn Dev data')
while True:
    try: _data = next(xn_dev_gen)
    except StopIteration: break
    xn_dev_set_to_dump['X'] += [_data[0]]
    xn_dev_set_to_dump['y'] += [_data[1]]
xn_dev_set_to_dump['X'] = np.concatenate(xn_dev_set_to_dump['X'])
xn_dev_set_to_dump['y'] = np.concatenate(xn_dev_set_to_dump['y'])

print('SG Dev data')
while True:
    try: _data = next(sg_dev_gen)
    except StopIteration: break
    sg_dev_set_to_dump['X'] += [_data[0]]
    sg_dev_set_to_dump['y'] += [_data[1]]
sg_dev_set_to_dump['X'] = np.concatenate(sg_dev_set_to_dump['X'])
sg_dev_set_to_dump['y'] = np.concatenate(sg_dev_set_to_dump['y'])


print('ID test data')
while True:
    try: _data = next(id_test_gen)
    except StopIteration: break
    id_test_set_to_dump['X'] += [_data[0]]
    id_test_set_to_dump['y'] += [_data[1]]
id_test_set_to_dump['X'] = np.concatenate(id_test_set_to_dump['X'])
id_test_set_to_dump['y'] = np.concatenate(id_test_set_to_dump['y'])

print('Xn data')
while True:
    try: _data = next(xn_test_gen)
    except StopIteration: break

    xn_test_set_to_dump['X'] += [_data[0]]
    xn_test_set_to_dump['y'] += [_data[1]]

xn_test_set_to_dump['X'] = np.concatenate(xn_test_set_to_dump['X'])
xn_test_set_to_dump['y'] = np.concatenate(xn_test_set_to_dump['y'])

print('SG data')
while True:
    try: _data = next(sg_test_gen)
    except StopIteration: break

    sg_test_set_to_dump['X'] += [_data[0]]
    sg_test_set_to_dump['y'] += [_data[1]]

sg_test_set_to_dump['X'] = np.concatenate(sg_test_set_to_dump['X'])
sg_test_set_to_dump['y'] = np.concatenate(sg_test_set_to_dump['y'])

print('Nomaly data')
while True:
    try: _data = next(anomaly_gen)
    except StopIteration: break

    anomaly_test_set_to_dump['X'] += [_data[0]]
    anomaly_test_set_to_dump['y'] += [_data[1]]

anomaly_test_set_to_dump['X'] = np.concatenate(anomaly_test_set_to_dump['X'])
anomaly_test_set_to_dump['y'] = np.array(anomaly_test_set_to_dump['y'])

viz_train = np.zeros((3, 28*9, 28*10), dtype='uint8')
for i in range(9):
    _pos = np.in1d(normal_training_set_to_dump['y'], i)
    this_X = normal_training_set_to_dump['X'][_pos]
    for j in range(10):
        viz_train[:, i*28:(i+1)*28, j*28:(j+1)*28] = this_X[j,...]

viz_id_dev = np.zeros((3, 28*9, 28*10), dtype='uint8')
for i in range(9):
    _pos = np.in1d(id_dev_set_to_dump['y'], i)
    this_X = id_dev_set_to_dump['X'][_pos]
    for j in range(10):
        viz_id_dev[:, i*28:(i+1)*28, j*28:(j+1)*28] = this_X[j,...]

viz_xn_dev = np.zeros((3, 28*9, 28*10), dtype='uint8')
for i in range(9):
    _pos = np.in1d(xn_dev_set_to_dump['y'], i)
    this_X = xn_dev_set_to_dump['X'][_pos]
    for j in range(10):
        viz_xn_dev[:, i*28:(i+1)*28, j*28:(j+1)*28] = this_X[j,...]

viz_sg_dev = np.zeros((3, 28*9, 28*10), dtype='uint8')
for i in range(9):
    _pos = np.in1d(sg_dev_set_to_dump['y'], i)
    this_X = sg_dev_set_to_dump['X'][_pos]
    for j in range(10):
        viz_sg_dev[:, i*28:(i+1)*28, j*28:(j+1)*28] = this_X[j,...]
        
viz_id_test = np.zeros((3, 28*9, 28*10), dtype='uint8')
for i in range(9):
    _pos = np.in1d(id_test_set_to_dump['y'], i)
    this_X = id_test_set_to_dump['X'][_pos]
    for j in range(10):
        viz_id_test[:, i*28:(i+1)*28, j*28:(j+1)*28] = this_X[j,...]

viz_xn = np.zeros((3, 28*9, 28*10), dtype='uint8')
for i in range(9):
    _pos = np.in1d(xn_test_set_to_dump['y'], i)
    this_X = xn_test_set_to_dump['X'][_pos]
    for j in range(10):
        viz_xn[:, i*28:(i+1)*28, j*28:(j+1)*28] = this_X[j,...]

viz_sg = np.zeros((3, 28*9, 28*10), dtype='uint8')
for i in range(9):
    _pos = np.in1d(sg_test_set_to_dump['y'], i)
    this_X = sg_test_set_to_dump['X'][_pos]
    for j in range(10):
        viz_sg[:, i*28:(i+1)*28, j*28:(j+1)*28] = this_X[j,...]

viz_ano = np.zeros((3, 28*9, 28*10), dtype='uint8')
for i in range(9):
    for j in range(10):
        viz_ano[:, i*28:(i+1)*28, j*28:(j+1)*28] = anomaly_test_set_to_dump['X'][i*9+j]

import imageio
imageio.imwrite('./images/train.png', np.transpose(viz_train, [1,2,0]))
imageio.imwrite('./images/id_dev.png', np.transpose(viz_id_dev, [1,2,0]))
imageio.imwrite('./images/xn_dev.png', np.transpose(viz_xn_dev, [1,2,0]))
imageio.imwrite('./images/sg_dev.png', np.transpose(viz_sg_dev, [1,2,0]))
imageio.imwrite('./images/test_id.png', np.transpose(viz_id_test, [1,2,0]))
imageio.imwrite('./images/test_xn.png', np.transpose(viz_xn, [1,2,0]))
imageio.imwrite('./images/test_sg.png', np.transpose(viz_sg, [1,2,0]))
imageio.imwrite('./images/test_anomaly.png', np.transpose(viz_ano, [1,2,0]))

pickle.dump(normal_training_set_to_dump, open(os.path.join(os.environ['DATA_DIR'], 'cmnist_train_{}.pickle'.format(_D)), 'wb'))
pickle.dump(id_dev_set_to_dump, open(os.path.join(os.environ['DATA_DIR'], 'cmnist_id_validation_{}.pickle'.format(_D)), 'wb'))
pickle.dump(xn_dev_set_to_dump, open(os.path.join(os.environ['DATA_DIR'], 'cmnist_xn_validation_{}.pickle'.format(_D)), 'wb'))
pickle.dump(sg_dev_set_to_dump, open(os.path.join(os.environ['DATA_DIR'], 'cmnist_sg_validation_{}.pickle'.format(_D)), 'wb'))
pickle.dump(id_test_set_to_dump, open(os.path.join(os.environ['DATA_DIR'], 'cmnist_id_test_{}.pickle'.format(_D)), 'wb'))
pickle.dump(xn_test_set_to_dump, open(os.path.join(os.environ['DATA_DIR'], 'cmnist_xn_test_{}.pickle'.format(_D)), 'wb'))
pickle.dump(sg_test_set_to_dump, open(os.path.join(os.environ['DATA_DIR'], 'cmnist_sg_test_{}.pickle'.format(_D)), 'wb'))
pickle.dump(anomaly_test_set_to_dump, open(os.path.join(os.environ['DATA_DIR'], 'cmnist_anomaly_{}.pickle'.format(_D)), 'wb'))
