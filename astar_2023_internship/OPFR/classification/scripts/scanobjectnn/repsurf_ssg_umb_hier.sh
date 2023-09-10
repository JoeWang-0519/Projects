#!/usr/bin/env bash
set -v

python3 train_cls_scanobjectnn_hier.py \
          --cuda_ops \
          --batch_size 64 \
          --model repsurf.repsurf_ssg_umb_hier \
          --epoch 300 \
          --log_dir repsurf_cls_ssg_umb_hier_modify2_trial2 \
          --gpus 1 \
          --n_workers 12 \
          --return_center \
          --return_dist \
          --return_polar \
          --group_size1 12 \
          --group_size2 2 \
          --group_size3 10 \
          --umb_pool1 max \
          --umb_pool2 sum \
          --num_point 1024
