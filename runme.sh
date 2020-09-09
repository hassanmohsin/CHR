#!/bin/bash
# select gpu devices
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m CHR.main -c configs/train_config.json |& tee -a log
