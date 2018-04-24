#!/bin/bash
dataset="/mnt/hdd/dataset/categorical_rgbd_flatten_split/"
resultdir=/mnt/hdd/depth2color/image_depth2rgb

for lam1 in 1;
do
  python train.py --gpu 0  --dataset $dataset --out ${resultdir}_lam1_$lam1_full --lam1 $lam1 --batchsize 20 --epoch 200
done
