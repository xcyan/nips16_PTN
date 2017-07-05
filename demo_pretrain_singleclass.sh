mkdir ./models
th scripts/train_rotatorRNN_base.lua --gpu 1 --niter 160 --save_every 40
th scripts/train_rotatorRNN_curriculum.lua --gpu 1 --kstep 2 --batch_size 32 --adam 1 --niter 40 --save_every 20
th scripts/train_rotatorRNN_curriculum.lua --gpu 1 --kstep 4 --batch_size 32 --adam 2 --niter 40 --save_every 20
th scripts/train_rotatorRNN_curriculum.lua --gpu 1 --kstep 8 --batch_size 32 --adam 2 --niter 40 --save_every 20
th scripts/train_rotatorRNN_curriculum.lua --gpu 1 --kstep 12 --batch_size 16 --adam 2 --niter 40 --save_every 20
th scripts/train_rotatorRNN_curriculum.lua --gpu 1 --kstep 16 --batch_size 8 --adam 2 --niter 40 --save_every 20

