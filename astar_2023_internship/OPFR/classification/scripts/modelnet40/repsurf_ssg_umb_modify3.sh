#!/usr/bin/env bash
set -v

python3 train_cls_modelnet40.py \
          --cuda_ops \
          --batch_size 32 \
          --model repsurf.repsurf_ssg_umb_modify3 \
          --epoch 200 \
          --log_dir repsurf_cls_ssg_umb_modify3_trial2 \
          --gpus 3 \
          --n_workers 12 \
          --return_center \
          --return_dist \
          --return_polar \
          --group_size 8 \
          --umb_pool sum \
          --num_point 1024 \
          --num_category 40
