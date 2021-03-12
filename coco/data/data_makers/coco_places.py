import os, sys, time, io, subprocess, requests
import numpy as np
import random
import pandas as pd

from PIL import Image
sys.path.append('/scratch/faruk/data/cocoapi/PythonAPI/')  # install cocoapi and change path here
from pycocotools.coco import COCO
from skimage.transform import resize

from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import h5py
################ Paths and other configs - Set these #################################
places_dir = os.path.join(os.environ['SLURM_TMPDIR'], 'data_256')
output_dir = '/scratch/faruk/data/cocoplaces/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

CLASSES = [
        'boat',
        'airplane',
        'truck',
        'dog',
        'zebra',
        'horse',
        'bird',
        'train',
        'bus'
        ]

NUM_CLASSES = len(CLASSES)
ANOMALIES = ['motorcycle']

biased_places = ['b/beach',
                 'c/canyon',
                 'b/building_facade',
                 's/staircase',
                 'd/desert/sand',
                 'c/crevasse',
                 'b/bamboo_forest',
                 'f/forest/broadleaf',
                 'b/ball_pit',
                 ]

unbiased_places = [
                 'k/kasbah',
                 'l/lighthouse',
                 'p/pagoda',
                 'r/rock_arch']

validation_unbiased_places = [
        'o/oast_house',
        'o/orchard',
        'v/viaduct']

test_unbiased_places = [
                 'w/water_tower',
                 'w/waterfall',
                 'z/zen_garden']

# copy over:
print('Copying (could take a while) ...', end='')
original_tarname = os.path.join(os.environ['DATA_DIR'], 'train_256_places365standard.tar')
if not os.path.exists(os.path.join(os.environ['SLURM_TMPDIR'], 'train_256_places365standard.tar')):
    subprocess.call(['cp', '-r', original_tarname, os.environ['SLURM_TMPDIR']])
print('Done!')

# extract 'em one by one
thing = os.path.join(os.environ['SLURM_TMPDIR'], 'train_256_places365standard.tar')
for place in biased_places:
    output = subprocess.check_output(['tar', '-xvf', thing, '-C', os.environ['SLURM_TMPDIR'], 'data_256/{}'.format(place)])
    print(output)

for place in unbiased_places:
    output = subprocess.check_output(['tar', '-xvf', thing, '-C', os.environ['SLURM_TMPDIR'], 'data_256/{}'.format(place)])
    print(output)

for place in validation_unbiased_places:
    output = subprocess.check_output(['tar', '-xvf', thing, '-C', os.environ['SLURM_TMPDIR'], 'data_256/{}'.format(place)])
    print(output)

for place in test_unbiased_places:
    output = subprocess.check_output(['tar', '-xvf', thing, '-C', os.environ['SLURM_TMPDIR'], 'data_256/{}'.format(place)])
    print(output)

print('Done!')

print('----- Bias places ------')
print(biased_places)

print('----- Unbias places ------')
print(unbiased_places)

print('----- Validation inbias places ------')
print(validation_unbiased_places)

print('----- Test unbias places ------')
print(test_unbiased_places)

confounder_strength = 0.8
dataset_name = 'cocoplaces_vf_{}_{}'.format(NUM_CLASSES, confounder_strength)
h5pyfname = os.path.join(output_dir, dataset_name)
print('---------------- FNAME --------------')
print(h5pyfname)
if not os.path.exists(h5pyfname):
    os.makedirs(h5pyfname)
######################################################################################

biased_place_fnames, validation_unbiased_place_fnames, unbiased_place_fnames, test_unbiased_place_fnames = {}, [], [], []
for i, target_place in enumerate(unbiased_places):
    L = [f'{target_place}/{filename}' for filename in os.listdir(os.path.join(places_dir, target_place)) if filename.endswith('.jpg')]
    unbiased_place_fnames += L
random.shuffle(unbiased_place_fnames)

for i, target_place in enumerate(validation_unbiased_places):
    L = [f'{target_place}/{filename}' for filename in os.listdir(os.path.join(places_dir, target_place)) if filename.endswith('.jpg')]
    validation_unbiased_place_fnames += L
random.shuffle(validation_unbiased_place_fnames)

for i, target_place in enumerate(biased_places):
    L = [f'{target_place}/{filename}' for filename in os.listdir(os.path.join(places_dir, target_place)) if filename.endswith('.jpg')]
    random.shuffle(L)
    biased_place_fnames[i] = L

for i, target_place in enumerate(test_unbiased_places):
    L = [f'{target_place}/{filename}' for filename in os.listdir(os.path.join(places_dir, target_place)) if filename.endswith('.jpg')]
    test_unbiased_place_fnames += L
random.shuffle(test_unbiased_place_fnames)

tr_i = 800*NUM_CLASSES
val_i = 100*NUM_CLASSES
te_i = 100*NUM_CLASSES

train_fname = os.path.join(h5pyfname,'train.h5py')

val_id_fname = os.path.join(h5pyfname,'validtest.h5py')
val_ood_fname = os.path.join(h5pyfname,'valoodtest.h5py')
val_sg_fname = os.path.join(h5pyfname,'valsgtest.h5py')

id_fname = os.path.join(h5pyfname,'idtest.h5py')
sg_fname = os.path.join(h5pyfname,'sgtest.h5py')
ood_fname =os.path.join( h5pyfname,'oodtest.h5py')

ano_fname =os.path.join( h5pyfname,'anotest.h5py')

if os.path.exists(train_fname): subprocess.call(['rm', train_fname])
if os.path.exists(val_id_fname): subprocess.call(['rm', val_id_fname])
if os.path.exists(val_ood_fname): subprocess.call(['rm', val_ood_fname])
if os.path.exists(val_sg_fname): subprocess.call(['rm', val_sg_fname])
if os.path.exists(id_fname): subprocess.call(['rm', id_fname])
if os.path.exists(sg_fname): subprocess.call(['rm', sg_fname])
if os.path.exists(ood_fname): subprocess.call(['rm', ood_fname])
if os.path.exists(ood_fname): subprocess.call(['rm', ano_fname])

train_file = h5py.File(train_fname, mode='w')
val_id_file = h5py.File(val_id_fname, mode='w')
val_ood_file = h5py.File(val_ood_fname, mode='w')
val_sg_file = h5py.File(val_sg_fname, mode='w')
id_test_file = h5py.File(id_fname, mode='w')
sg_test_file = h5py.File(sg_fname, mode='w')
ood_test_file = h5py.File(ood_fname, mode='w')
ano_test_file = h5py.File(ano_fname, mode='w')

train_file.create_dataset('images', (tr_i,3,64,64), dtype=np.dtype('float32'))
val_id_file.create_dataset('images', (val_i,3,64,64), dtype=np.dtype('float32'))
val_ood_file.create_dataset('images', (val_i,3,64,64), dtype=np.dtype('float32'))
val_sg_file.create_dataset('images', (val_i,3,64,64), dtype=np.dtype('float32'))
id_test_file.create_dataset('images', (te_i,3,64,64), dtype=np.dtype('float32'))
sg_test_file.create_dataset('images', (te_i,3,64,64), dtype=np.dtype('float32'))
ood_test_file.create_dataset('images', (te_i,3,64,64), dtype=np.dtype('float32'))
ano_test_file.create_dataset('images', (te_i,3,64,64), dtype=np.dtype('float32'))

train_file.create_dataset('y', (tr_i,), dtype='int32')
train_file.create_dataset('g', (tr_i,), dtype='int32')

val_id_file.create_dataset('y', (val_i,), dtype='int32')
val_id_file.create_dataset('g', (val_i,), dtype='int32')

val_ood_file.create_dataset('y', (val_i,), dtype='int32')
val_ood_file.create_dataset('g', (val_i,), dtype='int32')

val_sg_file.create_dataset('y', (val_i,), dtype='int32')
val_sg_file.create_dataset('g', (val_i,), dtype='int32')

id_test_file.create_dataset('y', (te_i,), dtype='int32')
id_test_file.create_dataset('g', (te_i,), dtype='int32')

sg_test_file.create_dataset('y', (te_i,), dtype='int32')
ood_test_file.create_dataset('y', (te_i,), dtype='int32')


coco = COCO('/scratch/faruk/data/annotations/instances_train2017.json')
cats = coco.loadCats(coco.getCatIds())

tr_s, val_s, te_s = 0, 0, 0

def getClassName(cID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == cID:
            return cats[i]['name']
    return 'None'

coco = COCO('/scratch/faruk/data/annotations/instances_train2017.json')
cats = coco.loadCats(coco.getCatIds())

print('Anomalies')
for c in range(len(ANOMALIES)):
    catIds = coco.getCatIds(catNms=[ANOMALIES[c]])
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    i = -1
    tr_si = 0
    print('Class {} (train) : #images = {}'.format(c, len(images)))
    while tr_si < tr_i//NUM_CLASSES:
        i += 1

        # get the image
        im = images[i]

        # get the annoatations
        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # pick largest area object
        max_ann = -1
        for _pos in range(len(anns)):
            if anns[_pos]['area'] > max_ann:
                pos = _pos
                max_ann = anns[_pos]['area']

        if max_ann < 10000: continue;

        try: img_data = requests.get(im['coco_url']).content
        except: time.sleep(10); img_data = requests.get(im['coco_url']).content
        I = np.asarray(Image.open(io.BytesIO(img_data)))
        if len(I.shape) == 2:
            I = np.tile(I[:,:,None], [1,1,3])

        # get the place
        idx = np.random.choice(range(NUM_CLASSES))
        place_path = biased_place_fnames[idx][tr_si]
        place_img = np.asarray(Image.open(os.path.join(places_dir, place_path)).convert('RGB'))

        # that's the one:
        mask = np.tile(255*coco.annToMask(anns[pos]).astype('uint8')[:,:,None], [1,1,3])
        resized_mask = resize(mask, (64, 64), anti_aliasing=True)

        resized_image = resize(I, (64, 64), anti_aliasing=True)
        resized_place = resize(place_img, (64, 64), anti_aliasing=True)

        new_im = resized_place*(1-resized_mask) + resized_image*resized_mask
        ano_test_file['images'][tr_si, ...] = np.transpose(new_im, (2,0,1))

        tr_si += 1
        if tr_si % 100 == 0:
            print('>'.format(c), end='')
            time.sleep(1)
    print('')

for c in range(NUM_CLASSES):
    catIds = coco.getCatIds(catNms=[CLASSES[c]])
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    i = -1
    tr_si = 0
    print('Class {} (train) : '.format(c), end=' ')
    while tr_si < tr_i//NUM_CLASSES:
        i += 1
        # get the image
        im = images[i]

        try: img_data = requests.get(im['coco_url']).content
        except: time.sleep(10); img_data = requests.get(im['coco_url']).content
        I = np.asarray(Image.open(io.BytesIO(img_data)))
        if len(I.shape) == 2:
            I = np.tile(I[:,:,None], [1,1,3])

        # pick largest area object
        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        max_ann = -1
        for _pos in range(len(anns)):
            if anns[_pos]['area'] > max_ann:
                pos = _pos
                max_ann = anns[_pos]['area']
        if max_ann < 10000: continue;

        # get the place
        if np.random.random() > confounder_strength:
            place_path = unbiased_place_fnames[tr_s]
            _g = 1
        else:
            place_path = biased_place_fnames[c][tr_si]
            _g = 0
        place_img = np.asarray(Image.open(os.path.join(places_dir, place_path)).convert('RGB'))

        # that's the one:
        mask = np.tile(255*coco.annToMask(anns[pos]).astype('uint8')[:,:,None], [1,1,3])

        resized_mask = resize(mask, (64, 64))
        resized_image = resize(I, (64, 64))
        resized_place = resize(place_img, (64, 64))

        new_im = resized_place*(1-resized_mask) + resized_image*resized_mask

        train_file['images'][tr_s, ...] = np.transpose(new_im, (2,0,1))
        train_file['y'][tr_s] = c
        train_file['g'][tr_s] = _g

        tr_s += 1
        tr_si += 1
        if tr_si % 100 == 0: print('>'.format(c), end='')
        ########################################
    print('')

    val_si = 0
    j = -1
    while val_si < val_i//NUM_CLASSES:
        i += 1
        j += 1

        # get the image
        im = images[i]

        # get the annoatations
        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # pick largest area object
        max_ann = -1
        for _pos in range(len(anns)):
            if anns[_pos]['area'] > max_ann:
                pos = _pos
                max_ann = anns[_pos]['area']

        if max_ann < 10000: continue;

        try: img_data = requests.get(im['coco_url']).content
        except: time.sleep(10); img_data = requests.get(im['coco_url']).content
        I = np.asarray(Image.open(io.BytesIO(img_data)))
        if len(I.shape) == 2:
            I = np.tile(I[:,:,None], [1,1,3])

        mask = np.tile(255*coco.annToMask(anns[pos]).astype('uint8')[:,:,None], [1,1,3])
        resized_mask = resize(mask, (64, 64), anti_aliasing=True)
        resized_image = resize(I, (64, 64), anti_aliasing=True)

        # val_id:
        if np.random.random() > confounder_strength:
            place_path = unbiased_place_fnames[j+int((1.0-confounder_strength)*tr_s)]
            _g = 1
        else:
            place_path = biased_place_fnames[c][j+int(confounder_strength*tr_si)]
            _g = 0
        place_img = np.asarray(Image.open(os.path.join(places_dir, place_path)).convert('RGB'))
        resized_place = resize(place_img, (64, 64), anti_aliasing=True)
        new_im = resized_place*(1-resized_mask) + resized_image*resized_mask

        val_id_file['images'][val_s, ...] = np.transpose(new_im, (2,0,1))
        val_id_file['y'][val_s] = c
        val_id_file['g'][val_s] = _g # doesn't mean what it meant

        # val_ood:
        place_path = validation_unbiased_place_fnames[val_s]
        place_img = np.asarray(Image.open(os.path.join(places_dir, place_path)).convert('RGB'))
        _g = 1
        resized_place = resize(place_img, (64, 64), anti_aliasing=True)
        new_im = resized_place*(1-resized_mask) + resized_image*resized_mask

        val_ood_file['images'][val_s, ...] = np.transpose(new_im, (2,0,1))
        val_ood_file['y'][val_s] = c
        val_ood_file['g'][val_s] = _g # doesn't mean what it meant

        # val_sg:
        idx = np.random.choice(list(set(range(NUM_CLASSES))-set([c])))
        place_path = biased_place_fnames[idx][j+int(confounder_strength*tr_si)]
        place_img = np.asarray(Image.open(os.path.join(places_dir, place_path)).convert('RGB'))
        _g = 0
        resized_place = resize(place_img, (64, 64), anti_aliasing=True)
        new_im = resized_place*(1-resized_mask) + resized_image*resized_mask

        val_sg_file['images'][val_s, ...] = np.transpose(new_im, (2,0,1))
        val_sg_file['y'][val_s] = c
        val_sg_file['g'][val_s] = _g # doesn't mean what it meant

        val_s += 1
        val_si += 1

    te_si = 0
    j = -1
    print('Class {} (test) : '.format(c), end=' ')
    while te_si < te_i//NUM_CLASSES:
        i += 1
        j += 1
        ########################################
        # In-dist test:
        # get the image
        im = images[i]

        try: img_data = requests.get(im['coco_url']).content
        except: time.sleep(10); img_data = requests.get(im['coco_url']).content
        I = np.asarray(Image.open(io.BytesIO(img_data)))
        if len(I.shape) == 2:
            I = np.tile(I[:,:,None], [1,1,3])

        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        max_ann = -1
        for _pos in range(len(anns)):
            if anns[_pos]['area'] > max_ann:
                pos = _pos
                max_ann = anns[_pos]['area']
        if max_ann < 10000: continue;

        # that's the one:
        mask = np.tile(255*coco.annToMask(anns[pos]).astype('uint8')[:,:,None], [1,1,3])
        resized_mask = resize(mask, (64, 64))
        resized_image = resize(I, (64, 64))

        # In-distribution:
        if np.random.random() > confounder_strength:
            place_path = unbiased_place_fnames[te_s+int((1.0-confounder_strength)*(tr_s+val_s))]
            _g = 1
        else:
            place_path = biased_place_fnames[c][te_si+int(confounder_strength*(tr_si+val_si))]
            _g = 0
        place_img = np.asarray(Image.open(os.path.join(places_dir, place_path)).convert('RGB'))
        resized_place = resize(place_img, (64, 64))

        new_im = resized_place*(1-resized_mask) + resized_image*resized_mask

        id_test_file['images'][te_s, ...] = np.transpose(new_im, (2,0,1))
        id_test_file['y'][te_s] = c
        id_test_file['g'][te_s] = _g

        # Out-of-distribution:
        place_path = test_unbiased_place_fnames[j]
        place_img = np.asarray(Image.open(os.path.join(places_dir, place_path)).convert('RGB'))
        resized_place = resize(place_img, (64, 64))

        new_im = resized_place*(1-resized_mask) + resized_image*resized_mask

        ood_test_file['images'][te_s, ...] = np.transpose(new_im, (2,0,1))
        ood_test_file['y'][te_s] = c

        # Systematic generalisation:
        idx = np.random.choice(list(set(range(NUM_CLASSES))-set([c])))
        place_path = biased_place_fnames[idx][j+int(confounder_strength*(tr_si+val_si))]
        place_img = np.asarray(Image.open(os.path.join(places_dir, place_path)).convert('RGB'))
        resized_place = resize(place_img, (64, 64))

        new_im = resized_place*(1-resized_mask) + resized_image*resized_mask

        sg_test_file['images'][te_s, ...] = np.transpose(new_im, (2,0,1))
        sg_test_file['y'][te_s] = c

        te_s += 1
        te_si += 1
        if te_si % 100 == 0: print('>'.format(c), end='')
        ########################################
    print('')

train_file.close()
val_id_file.close()
val_ood_file.close()
val_sg_file.close()
id_test_file.close()
sg_test_file.close()
ood_test_file.close()
ano_test_file.close()
