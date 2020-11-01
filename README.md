# Perspective Transformer Nets (PTN)

This is the code for NIPS 2016 paper [Perspective Transformer Nets: Learning Single-View 3D Object Reconstruction without 3D Supervision](https://papers.nips.cc/paper/6206-perspective-transformer-nets-learning-single-view-3d-object-reconstruction-without-3d-supervision.pdf) by Xinchen Yan, Jimei Yang, Ersin Yumer, Yijie Guo and Honglak Lee

<img src="https://b191c0a7-a-62cb3a1a-s-sites.googlegroups.com/site/skywalkeryxc/perspective_transformer_nets/website_background.png" width="900px" height="300px"/>

Please follow the instructions to run the code.

## Requirements
PTN requires or works with 
* Mac OS X or Linux
* NVIDIA GPU

## Installing Dependency
* Install [Torch](http://torch.ch)
* Install [Mattorch](https://github.com/clementfarabet/lua---mattorch)
* Install [Perspective Transformer Layer](https://github.com/xcyan/ptnbhwd.git)

The following command installs the Perspective Transformer Layer:
```
./install_ptnbhwd.sh
```

## Dataset Downloading
* Please run the command to download the pre-processed dataset (including rendered 2D views and 3D volumes):
```
./prepare_data.sh
```
* Disclaimer: Please cite the [ShapeNet paper](https://arxiv.org/pdf/1512.03012.pdf) as well.

## Pre-trained Models Downloading (single-class experiment)

PTN-Proj: ptn_proj.t7

PTN-Comb: ptn_comb.t7

CNN-Vol: cnn_vol.t7

* The following command downloads the pre-trained models:
```
./download_models.sh
```

## Testing using Pre-trained Models (single-class experiment)

* The following command evaluates the pre-trained models:
```
./eval_models.sh
```

## Training (single-class experiment)
* If you want to pre-train the view-point indepedent image encoder on single-class, please run the following command.
Note that the pre-training could take a few days on a single TITAN X GPU.
```
./demo_pretrain_singleclass.sh
```
* If you want to train PTN-Proj (unsupervised) on single-class based on pre-trained encoder, please run the command.
```
./demo_train_ptn_proj_singleclass.sh
```
* If you want to train PTN-Comb (3D supervision) on single-class based on pre-trained encoder, please run the command.
```
./demo_train_ptn_comb_singleclass.sh
```
* If you want to train CNN-Vol (3D supervision) on single-class based on pre-trained encoder, please run the command.
```
./demo_train_cnn_vol_singleclass.sh
```

## Using your own camera
* In many cases, you want to implement your own camera matrix (e.g., intrinsic or extrinsic). 
Please feel free to modify [this function](https://github.com/xcyan/nips16_PTN/blob/master/scripts/train_PTN.lua#L207).

* Before start your own implementation, we recommand to go through some basic camera geometry in [this computer vision textbook](http://szeliski.org/Book/drafts/SzeliskiBook_20100903_draft.pdf) written by Richard Szeliski (see Eq 2.59 at Page 53).

* Note that in our voxel ray-tracing implementation, we used the inverse camera matrix. 

## Third-party Implementation

Besides our torch implementation, we recommend to see also the following third-party re-implementation:
* [TensorFlow Implementation](https://github.com/tensorflow/models/tree/archive/research/ptn): This re-implementation was developed during Xinchen's Google internship; If you find a bug, please file a bug including @xcyan. 

## Citation

If you find this useful, please cite our work as follows:
```
@incollection{NIPS2016_6206,
title = {Perspective Transformer Nets: Learning Single-View 3D Object Reconstruction without 3D Supervision},
author = {Yan, Xinchen and Yang, Jimei and Yumer, Ersin and Guo, Yijie and Lee, Honglak},
booktitle = {Advances in Neural Information Processing Systems 29},
editor = {D. D. Lee and M. Sugiyama and U. V. Luxburg and I. Guyon and R. Garnett},
pages = {1696--1704},
year = {2016},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/6206-perspective-transformer-nets-learning-single-view-3d-object-reconstruction-without-3d-supervision.pdf}
}
```
