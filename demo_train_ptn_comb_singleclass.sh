mkdir ./models
th scripts/train_PTN.lua --gpu 2 --niter 100 --lambda_vox 1 --lambda_msk 1

