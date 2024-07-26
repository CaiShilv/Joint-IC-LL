#!/bin/bash
OMP_NUM_THREADS=4 python -m torch.distributed.launch --use_env --master_port=6167 --nproc_per_node=4 train.py -opt=./options/train/all_data_combine_psnr.yml --launcher=pytorch