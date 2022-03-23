#! /bin/bash

source /ocean/projects/asc170022p/yanwuxu/miniconda/etc/profile.d/conda.sh
conda activate crossMoDA

direction='AtoB'
dataroot='../data/cityscapes'
batch_size=4
load_size=128
crop_size=128

netG=resnet_9blocks
netD=basic

python train.py --dataroot $dataroot --model mt_gan \
--pool_size 50 --no_dropout --load_size $load_size --crop_size $crop_size \
--netG $netG --netD $netD --batch_size $batch_size \
--direction $direction;
python test.py --dataroot $dataroot --model mt_gan  --eval \
--no_dropout --load_size $load_size --crop_size $crop_size \
--netG $netG --netD $netD --batch_size $batch_size \
--direction $direction
