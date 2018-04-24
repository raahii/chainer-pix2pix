#!/bin/bash
set -e
dataset="/mnt/hdd/dataset/categorical_rgbd_flatten_split/"
resultdir=/mnt/hdd/depth2color/image_depth2rgb

for lam1 in 100 75 50 25 1;
do
  python train.py --gpu 0  --dataset $dataset --out ${resultdir}_lam1_$lam1 --lam1 $lam1 --batchsize 50 --epoch 20
done
