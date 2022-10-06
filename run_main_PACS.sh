#!/usr/bin/env bash

max=1

for i in $(seq 1 $max); do
  python train.py \
    --task='PACS' \
    --seed=1\
    --alpha1=1\
    --alpha2=1\
    --alpha3=0.0001\
    --beta=0.1\
    --lr_sc=0.005\
    --epochs=50\
    --learning_rate=0.005\
    --ours
done
