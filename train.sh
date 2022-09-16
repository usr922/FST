#!/bin/bash
g=$(($1<8?$1:8))
now=$(date +"%Y%m%d_%H%M%S")
export PYTHONPATH=$ROOT:$PYTHONPATH

# resnet-101
python run_experiments.py --config configs/fst-d/gta2cs_uda_warm_fdthings_rcs_croppl_a999_nesterov_r101_s0_step3.py