#!/bin/bash

#SBATCH --job-name=deepMNIST
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --output=mymnist.%j.log

# load modules
module use /apps/daint/UES/6.0.UP02/sandbox-dl/modules/all
module load daint-gpu
module load TensorFlow/1.2.1-CrayGNU-17.08-cuda-8.0-python3

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-daint/bin/activate

# run MNIST
# cd $HOME/mymnist
srun python3 deepMNIST_gpu.py \
--batch_size=50 \
--train_steps=20000 \
--data_format=NHWC \
--display_every=100 \
--data_dir=./MNIST_data 

# deactivate virtualenv
deactivate
