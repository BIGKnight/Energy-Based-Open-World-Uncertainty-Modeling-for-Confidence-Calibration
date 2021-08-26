#!/bin/bash
python train.py \
--lr .0001 \
--data_root ${1} \
--dataset cifar100 \
--optimizer adam \
--sigma ${4} \
--width 10 \
--depth 28 \
--save_dir ../checkpoints/wsrnet-8-17 \
--warmup_iters 1000 \
--gpu 0 \
--ebm_modify \
--platt_scaling \
--n_epochs ${3} \
--ebm_start_epoch ${2} \
${@:5}