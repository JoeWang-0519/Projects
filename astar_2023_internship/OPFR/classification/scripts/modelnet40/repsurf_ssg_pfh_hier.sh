#!/usr/bin/env bash
set -v

python3 train_cls_modelnet40_hier.py \
          --cuda_ops \
          --batch_size 64 \
          --model repsurf.repsurf_ssg_pfh_hier \
          --epoch 200 \
          --log_dir repsurf_cls_ssg_pfh_hier_modify1_trial2 \
          --gpus 3 \
          --n_workers 12 \
          --return_center \
          --return_dist \
          --return_polar \
          --group_size1 40 \
          --group_size2 3 \
          --group_size3 8 \
          --umb_pool1 max \
          --umb_pool2 avg \
          --num_point 1024 \
          --num_category 40
