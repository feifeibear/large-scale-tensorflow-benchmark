#!/bin/bash

#SBATCH --job-name=dist_deepMNIST
#SBATCH --time=3:30:00
#SBATCH --nodes=9
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
export WORK_DIR=`pwd`/..
export TF_SCRIPT="${WORK_DIR}/resnet_cifar_main.py"
export TF_EVAL_SCRIPT="${WORK_DIR}/resnet_cifar_eval.py"
export DATASET=cifar10

export TF_FLAGS="
  --train_data_path=${SCRATCH}/data/cifar-10-batches-bin/data_batch* \
  --log_root=./tmp/resnet_model \
  --train_dir=./tmp/resnet_model/train \
  --dataset=${DATASET} \
  --num_gpus=1 \
  --batch_size=16 \
  --sync_replicas=True \
  --train_steps=80000
"

export TF_EVAL_FLAGS="
  --eval_data_path=${SCRATCH}/data/cifar-10-batches-bin/test_batch* \
  --log_root=./tmp/resnet_model \
  --eval_dir=./tmp/resnet_model/test \
  --dataset=${DATASET} \
  --mode=eval \
  --num_gpus=1
"


# set TensorFlow distributed parameters
export TF_NUM_PS=$1 # 1
export TF_NUM_WORKERS=$2 # $SLURM_JOB_NUM_NODES
# export TF_WORKER_PER_NODE=1
# export TF_PS_PER_NODE=1
# export TF_PS_IN_WORKER=true

# run distributed TensorFlow
DIST_TF_LAUNCHER_SCRIPT=run_dist_train_eval_daint.sh
DIST_TF_LAUNCHER_DIR=./logs/$1-ps-$2-wk-${DATASET}-log #$SCRATCH/run_dist_tf_daint_directory
rm -rf $DIST_TF_LAUNCHER_DIR/*.log
rm -rf $DIST_TF_LAUNCHER_DIR/*.sh
mkdir -p $DIST_TF_LAUNCHER_DIR
cp ${DIST_TF_LAUNCHER_SCRIPT} $DIST_TF_LAUNCHER_DIR
cd $DIST_TF_LAUNCHER_DIR
./${DIST_TF_LAUNCHER_SCRIPT}

# deactivate virtualenv
deactivate
