import os, sys, glob, time, subprocess
import h5py
import numpy as np

from PIL import Image

def get_handles(num_classes=2, confounder_strength=0.8, dataset='colour', validation=True):
    if dataset == 'places':
        dataset_name = 'cocoplaces_vf_{}_{}'.format(num_classes, confounder_strength)
        original_dirname = os.path.join('/home/mila/a/ahmedfar/data', dataset_name)
    elif dataset == 'colour':
        dataset_name = 'cococolours_vf_{}_{}'.format(num_classes, confounder_strength)
        original_dirname = os.path.join('/home/mila/a/ahmedfar/data', dataset_name)

    dirname = os.path.join(os.environ['SLURM_TMPDIR'],  dataset_name)

    print('Copying data over, this will be worth it, be patient ...', end=' ')
    subprocess.call(['rsync', '-r', original_dirname, os.environ['SLURM_TMPDIR']])
    print('Done!')

    train_file = h5py.File(dirname+'/train.h5py', mode='r')

    if validation:
        id_test_file = h5py.File(dirname+'/validtest.h5py', mode='r')
        ood_test_file = h5py.File(dirname+'/valoodtest.h5py', mode='r')
        sg_test_file = h5py.File(dirname+'/valsgtest.h5py', mode='r')
    else:
        id_test_file = h5py.File(dirname+'/idtest.h5py', mode='r')
        ood_test_file = h5py.File(dirname+'/oodtest.h5py', mode='r')
        sg_test_file = h5py.File(dirname+'/sgtest.h5py', mode='r')

    ano_test_file = h5py.File(dirname+'/anotest.h5py', mode='r')

    return (train_file, id_test_file, ood_test_file, sg_test_file, ano_test_file)


def get_train_batch(handle, inds, anomaly=None):
    images = handle['images'][inds]
    labels = handle['y'][inds]
    groups = handle['g'][inds]
    return (images, labels, groups)


def get_eval_batch(handle, bs, N):
    inds = sorted(list(np.random.choice(N, bs, replace=False)))
    images = handle['images'][inds]
    try:
        labels = handle['y'][inds]
    except KeyError:
        labels = None
    return (images, labels)


def in_sequence_train_batch(handle, N):
    for i in range(N):
        images = handle['images'][i:i+1, ...]
        labels = handle['y'][i:i+1]
        groups = handle['g'][i:i+1]
        yield (images, labels, groups)


def get_eval_iterator(handle, N, batch_size, return_groups=False):
    assert N%batch_size == 0
    for i in range(N//batch_size):
        inds = slice(i*batch_size, (i+1)*batch_size)
        images = handle['images'][inds]
        try:
            labels = handle['y'][inds]
        except KeyError: labels = None
        if return_groups:
            groups = handle['g'][inds]
        else:
            groups = None
        yield (images, labels, groups)
