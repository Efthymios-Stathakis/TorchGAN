#!/bin/bash

# Save this script as run_dcgan.sh and give it execute permissions
# Usage: ./run_dcgan.sh [n_epochs] [z_dim] [batch_size] [hidden_dim] [lr] [beta_1] [beta_2]

n_epochs=${1:-50}
z_dim=${2:-128}
batch_size=${3:-128}
hidden_dim=${4:-64}
lr=${5:-0.0002}
beta_1=${6:-0.5}
beta_2=${7:-0.999}

python -m train.py --n_epochs $n_epochs \
                   --z_dim $z_dim \
                   --batch_size $batch_size \
                   --hidden_dim $hidden_dim \
                   --lr $lr \
                   --beta_1 $beta_1 \
                   --beta_2 $beta_2