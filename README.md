# Systematic generalisation with group invariant predictions

Code to accompany https://openreview.net/pdf?id=b9PoimzZFJ.

### Requirements and usage
Requirements are Python 3, TensorFlow v1.14, Numpy, Scipy, Scikit-Learn, Matplotlib, Pillow, Scikit-Image, h5py, tqdm. Experiments were run on V100 GPUs (16 and 32GB).

The `mnist` folder contains code used to run the Coloured-MNIST experiments. The `data` folder in it contains data generators and includes a sample dataset generation. The inferred partitions that were used are also included in `predicted_partitions`.

For example, in the `mnist` folder, you could run
```
python main.py -method pgi -wc 50.0 -u 5
```
Use the `-v` flag when performing validation.

For `coco` you can create the datasets using the code in `data_makers`, which will require installing the [cocoapi](https://github.com/cocodataset/cocoapi) and downloading the [Places](http://places2.csail.mit.edu/) dataset. A sample of generated COCO datasets can be downloaded [here](https://www.dropbox.com/sh/gn99k2pllwoot87/AAAk3nxlbbqvAX8DhA7Dr9Tma?dl=0).


For example, in the `coco` folder, you could run
```
python main.py -dataset colour -method pgi -wc 100.0 -u 200
python main.py -dataset places -sample normal -method pgi -wc 50.0 -u 200
```

### Citation
BibTex:
```
@proceedings{ahmed2021systematic,
  title={Systematic generalisation with group invariant predictions},  
  author={Ahmed, Faruk and Bengio, Yoshua and van Seijen, Harm and Courville, Aaron},  
  booktitle={9th International Conference on Learning Representations (ICLR)},  
  year={2021}  
}
```
