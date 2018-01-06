#!/bin/bash

#SBATCH --job-name=dist_google_benchmark
#SBATCH --time=00:30:00
#SBATCH --nodes=128
#SBATCH --constraint=gpu
#SBATCH --output=dist_benchmark_daint.%j.log

# Arguments:
#   $1: TF_NUM_PS: number of parameter servers
#   $2: TF_NUM_WORKER: number of workers
#   $3: variable_update: parameter_server/distributed_replicated
#   $4: real_data: true/false

# load modules
module use /apps/daint/UES/6.0.UP02/sandbox-dl/modules/all
module load daint-gpu
module load TensorFlow/1.2.1-CrayGNU-17.08-cuda-8.0-python3
#module load python
#export PYTHONPATH=/usr/common/software/tensorflow/intel-tensorflow/head/lib/python2.7/site-packages

# load virtualenv
export WORKON_HOME=~/Envs
source $WORKON_HOME/tf-daint/bin/activate

# set TensorFlow script parameters
export ROOT_DIR=/scratch/snx3000/youyang9/fjr/tf_workspace/large-scale-tensorflow-benchmark
export TF_SCRIPT=$ROOT_DIR/google-benchmarks/tf_cnn_benchmarks/tf_cnn_benchmarks.py

data_flags="
--num_gpus=1 \
--device=cpu \
--batch_size=32 \
--data_format=NHWC \
--variable_update=$3 \
--local_parameter_device=cpu \
--optimizer=sgd \
--model=$5 \
--data_name=imagenet \
--data_dir=/scratch/snx3000/maximem/deeplearnpackages/ImageNet/TF/
"
nodata_flags="
--num_gpus=1 \
--device=gpu \
--batch_size=32 \
--data_format=NHWC \
--variable_update=$3 \
--local_parameter_device=cpu \
--optimizer=sgd \
--model=$5 \
--data_name=imagenet
"

if [ "$4" = "true" ]; then
  export TF_FLAGS=$data_flags
elif [ "$4" = "false" ]; then
  export TF_FLAGS=$nodata_flags
else
  echo "error in real_data argument"
  exit 1
fi

# set TensorFlow distributed parameters
export TF_NUM_PS=$1
export TF_NUM_WORKERS=$2 # $SLURM_JOB_NUM_NODES
# export TF_WORKER_PER_NODE=1
# export TF_PS_PER_NODE=1
# export TF_PS_IN_WORKER=true

# run distributed TensorFlow
DIST_TF_LAUNCHER_DIR=$SCRATCH/fjr/tf/tf-$1-$2-$3-$4-$5
cd $DIST_TF_LAUNCHER_DIR
rm -rf .tfdist* worker.* ps.*
./run_dist_tf.sh

# deactivate virtualenv
deactivate
