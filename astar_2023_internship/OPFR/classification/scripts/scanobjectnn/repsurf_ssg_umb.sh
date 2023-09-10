#!/usr/bin/env bash
set -v

python3 train_cls_scanobjectnn.py \
          --cuda_ops \
          --batch_size 64 \
          --model repsurf.repsurf_ssg_umb \
          --epoch 250 \
          --log_dir repsurf_cls_ssg_umb_pfhonly_modify3_trial2 \
          --gpus 1 \
          --n_workers 12 \
          --return_center \
          --return_dist \
          --return_polar \
          --group_size 8 \
          --umb_pool sum \
          --num_point 1024 \
	  --use_normals \
	  --use_pfh
