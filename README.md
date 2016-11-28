# Perspective Transformer Nets (PTN)

This is the code for NIPS 2016 paper [Perspective Transformer Nets: Learning Single-View 3D Object Reconstruction without 3D Supervision](https://papers.nips.cc/paper/6206-perspective-transformer-nets-learning-single-view-3d-object-reconstruction-without-3d-supervision.pdf) by Xinchen Yan, Jimei Yang, Ersin Yumer, Yijie Guo and Honglak Lee

Please follow the instructions to run the code.

## Requirements
PTN requires or works with 
* Mac OS X or Linux
* NVIDIA GPU

## Installing Dependency
* Install [Torch](http://torch.ch)
* Install [Torch Perspective Transformer Layer](https://github.com/xcyan/ptnbhwd.git)
```
./install_ptnbhwd.sh
```

## Dataset Downloading
* Please run the script to download the pre-processed dataset (including rendered 2D views and 3D volumes):
```
./prepare_data.sh
```

## Training (single-class experiment)
* If you want to pre-train the view-point indepedent image encoder on single-class, please run the script
```
./demo_pretrain_singleclasss.sh
```
* If you want to train PTN-Proj (unsupervised) on single-class based on pre-trained encoder, please run the script
```
./demo_train_ptn_proj_singleclass.sh
```
* If you want to train PTN-Comb (3D supervision) on single-class based on pre-trained encoder, please run the script
```
./demo_train_ptn_comb_singleclass.sh
```
* If you want to train CNN-Vol (3D supervision) on single-class based on pre-trained encoder, please run the script
```
./demo_train_cnn_vol_singleclass.sh
```

## Testing using Pre-trained Model (single-class experiment)
TBD

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
