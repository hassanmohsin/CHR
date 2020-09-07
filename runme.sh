#!/bin/bash
# select gpu devices
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# python -m CHR.main --batch-size 320 --epochs 15 --workers 56 --data /data2/mhassan/dhs/datasets/sixray/dataset --model_path ./models_100 |& tee -a log_100


# DON'T FORGET TO CHANGE THE RAY.PY TO 10/100/1000

# Chanti00
python -m CHR.main --batch-size 128 --epochs 15 --workers 24 --resume ./models_1000/checkpoint.pth.tar --data /data/hassan/dhs/datasets/sixray/dataset --model_path ./models_1000 |& tee -a log_1000

# Chanti01
#python -m CHR.main --batch-size 320 --epochs 15 --workers 56 --resume ./models_100/checkpoint.pth.tar --data /data2/mhassan/dhs/datasets/sixray/dataset --model_path ./models_100 |& tee -a log_100
# python -m CHR.main --batch-size 320 --epochs 15 --workers 56 --data /data2/mhassan/dhs/datasets/sixray/dataset --model_path ../models |& tee -a log_10
