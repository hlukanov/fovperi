# Biologically inspired deep learning model for efficient foveal-peripheral vision

Code accompanying the paper "Biologically inspired deep learning model for efficient foveal-peripheral vision" by Hristofor Lukanov, Peter König and Gordon Pipa, Institute of Cognitive science, University of Osnabrück.

![](https://raw.githubusercontent.com/hlukanov/fovperi/main/readme_images/model.png)

## Requirements:
- Python 3.6
- tensorflow-gpu 1.14.0
- Keras 2.2.4
- numpy 1.15.4
- SharedArray 3.2.1
- albumentations 0.4.5
- opencv-python 4.2.0.32
- h5py 2.9.0

## Overview

This repository consists of four main folders:
- `fov` - contains tools for creating a sampling matrix, as well as precompiled mappings (forward and reverse) for foveation
- `preprocessing` contains tools for preprocessing the datasets
- `weights` contains weights for models trained on imagenet and core50
- `utils` contains tools for performing foveation and utilities helpful for training and testing

## Datasets and data preparation
The dataset needs to be in `*.hdf5` format and be named `imagenet.hdf5` or `core50.hdf5`.

The structure of the hdf5 file is the following:
- Key `train/images` includes the train image set with shape `(TOTAL_IMAGES,H,W,C)` in `uint8` datatype, where H=W=256 and C=3.
`TOTAL_IMAGES` is the total number of images in the dataset, `H` and `W` stand for respectively height and width, and `C` stands for color channels.
- Key `train/labels` contains the labels for training data for ILSVRC'10 in onehot format. `train/labels10`, `train/labels50` respectively contain labels with 10 classes or 50 classes for Core50.
- Key `test/images` is identical to `train/images` but with the test set.
- Keys `test/labels`, `test/labels10` and `test/labels50` are identical to their training counterparts but for test data.

### ILSVRC 2010
The dataset can be downloaded from [here](https://www.image-net.org/challenges/LSVRC/2010/index.php "here"). The validation set in ILSVRC 2010 is directly downloaded from the source page.

All images are resized such that the smaller dimension equals 256 pixels and cropped to a 256x256 pixel square images.

### Core50
Core50 can be downloaded from [here](https://vlomonaco.github.io/core50/ "here").

The test set of Core50 contains the following folders:
```
s1: o47, o17, o6, o13, o28, o37, o42, o35
s2: o43, o39, o42, o25, o46, o13, o22, o34
s3: o50, o21, o47, o28, o15, o3, o49, o9
s4: o39, o3, o38, o26, o30, o5, o33, o29
s5: o28, o31, o37, o3, o11, o10, o23
s6: o14, o17, o2, o21, o34, o35, o18, o42
s7: o36, o7, o22, o45, o12, o4, o49
s8: o19, o34, o12, o20, o15, o38
s9: o29, o16, o19, o18, o24, o31
s10: o10, o48, o25, o27, o44, o38, o49
s11: o25, o9, o27, o19, o18, o40, o7
```

All images are resized to 256x256 pixel size.

## How to train and test the model

To train execute:
```
python main.py --model=[imagenet|core50] --task=train
```

To test execute:
```
python main.py --model=[imagenet|core50] --task=test
```

## License
*[to be added]*

## Cite the paper
```
@article{lukanov2021biologically,
  title={Biologically Inspired Deep Learning Model for Efficient Foveal-Peripheral Vision},
  author={Lukanov, Hristofor and K{\"o}nig, Peter and Pipa, Gordon},
  journal={Frontiers in Computational Neuroscience},
  volume={15},
  pages={746204},
  year={2021},
  publisher={Frontiers Media SA}
}
```
