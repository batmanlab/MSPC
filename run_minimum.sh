#! /bin/bash

source /ocean/projects/asc170022p/yanwuxu/miniconda/etc/profile.d/conda.sh
conda activate crossMoDA

python train.py --dataroot ../data/horse2zebra/ --model minimum_gan \
--lambda_minimum 0.1 --load_size 286 --crop_size 256 \
--batch_size 4 --num_threads 16 --pool_size 50 \
--netG resnet_6blocks --no_antialias --no_antialias_up \
--master_port 69290 --world_size 1 --rank 0
