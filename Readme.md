# EOW-Softmax
This code is for the paper ["Energy-Based Open-World Uncertainty Modeling for Confidence Calibration"](https://arxiv.org/abs/2107.12628).
## Usage
    python train.py \
    --lr .0001 \
    --dataset cifar100 \
    --optimizer adam \
    --sigma 0.01 \
    --width 10 \
    --depth 28 \
    --warmup_iters 1000 \
    --ebm_modify \
    --n_epochs 15 \
    --ebm_start_epoch 8 \
    --ebm_weight 0.1
