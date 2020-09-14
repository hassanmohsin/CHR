#!/bin/bash
# select gpu devices
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# train
# python -m chr.main -c configs/train_config.json |& tee -a log

# test
python -m chr.main -c configs/test_config.json |& tee -a test.log
