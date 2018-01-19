#!/bin/bash

#SBATCH --job-name=resnet
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --output=resnet.%j.log

# load modules
module use /apps/daint/UES/6.0.UP02/sandbox-dl/modules/all
module load daint-gpu
module load TensorFlow/1.2.1-CrayGNU-17.08-cuda-8.0-python3

srun -n 1 python3 input_test.py
