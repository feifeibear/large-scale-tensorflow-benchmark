#!/bin/bash

#SBATCH --job-name=dist_deepMNIST
#SBATCH --time=00:60:00
#SBATCH --nodes=16
#SBATCH --constraint=gpu
#SBATCH --output=dist_cifar.%j.log

# Arguments:
#   $1: TF_NUM_PS: number of parameter servers
#   $2: TF_NUM_WORKER: number of workers

# load modules
module use /apps/daint/UES/6.0.UP02/sandbox-dl/modules/all
module load daint-gpu
#module load TensorFlow/1.2.1-CrayGNU-17.08-cuda-8.0-python3
module load TensorFlow/1.3.0-CrayGNU-17.08-cuda-8.0-python3

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-daint/bin/activate

# set TensorFlow script parameters
# export TF_SCRIPT="$HOME/mymnist/dist_deepMNIST_gpu.py"
export TF_SCRIPT="/scratch/snx3000/youyang9/fjr/tf_workspace/large-scale-tensorflow-benchmark/resnetv2/resnet_imagenet_main.py"
export WORK_DIR="/scratch/snx3000/youyang9/fjr/tf_workspace/large-scale-tensorflow-benchmark/resnetv2"
export DATASET=imagenet

export TF_FLAGS="
  --train_data_path=${SCRATCH}/data/imagenet \
  --log_root=./tmp/resnet_model \
  --train_dir=./tmp/resnet_model/train \
  --dataset=${DATASET} \
  --num_gpus=1 \
  --batch_size=32 \
  --sync_replicas=True \
  --train_steps=100
"

# set TensorFlow distributed parameters
export TF_NUM_PS=$1 # 1
export TF_NUM_WORKERS=$2 # $SLURM_JOB_NUM_NODES
# export TF_WORKER_PER_NODE=1
# export TF_PS_PER_NODE=1
# export TF_PS_IN_WORKER=true

# run distributed TensorFlow
DIST_TF_LAUNCHER_DIR=./logs/$1-ps-$2-wk-${DATASET}-log #$SCRATCH/run_dist_tf_daint_directory
rm -rf $DIST_TF_LAUNCHER_DIR
mkdir -p $DIST_TF_LAUNCHER_DIR
cp run_dist_tf_daint.sh $DIST_TF_LAUNCHER_DIR
cd $DIST_TF_LAUNCHER_DIR
./run_dist_tf_daint.sh

# deactivate virtualenv
deactivate
