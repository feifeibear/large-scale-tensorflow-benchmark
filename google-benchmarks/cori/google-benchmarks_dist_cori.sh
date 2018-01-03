#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 8 
#SBATCH -C knl,quad,cache
#SBATCH -t 00:12:00
#SBATCH -L SCRATCH
#SBATCH -J google_benchmark 
#SBATCH --output=dist_inception.%j.log

# Arguments:
#   $1: TF_NUM_PS: number of parameter servers
#   $2: TF_NUM_WORKER: number of workers
#   $3: variable_update: parameter_server/distributed_replicated
#   $4: real_data: true/false

# load modules
module load tensorflow/intel-head
#module load tensorflow/1.4.0rc0
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=66
KMP_AFFINITY="granularity=fine,noverbose,compact,1,0"
KMP_SETTINGS=1
KMP_BLOCKTIME=1

# load virtualenv
#export WORKON_HOME=~/Envs
#source $WORKON_HOME/tf-daint/bin/activate

# set TensorFlow script parameters
export TF_SCRIPT="/global/cscratch1/sd/yyang420/fjr/tensorflow/distributed-tensorflow-benchmarks/google-benchmarks/tf_cnn_benchmarks/tf_cnn_benchmarks.py"

data_flags="
--num_gpus=1 \
--device=cpu \
--batch_size=64 \
--data_format=NHWC \
--variable_update=$3 \
--local_parameter_device=cpu \
--optimizer=sgd \
--model=inception3 \
--data_name=imagenet \
--data_dir=/scratch/snx3000/maximem/deeplearnpackages/ImageNet/TF/
"
nodata_flags="
--num_gpus=1 \
--device=cpu \
--batch_size=64 \
--data_format=NHWC \
--variable_update=$3 \
--local_parameter_device=cpu \
--optimizer=sgd \
--model=inception3 \
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
export TF_PS_IN_WORKER=true
# export TF_WORKER_PER_NODE=1
# export TF_PS_PER_NODE=1
# export TF_PS_IN_WORKER=true

# run distributed TensorFlow
#DIST_TF_LAUNCHER_DIR=$SCRATCH/fjr/tf
#cd $DIST_TF_LAUNCHER_DIR
#current_time=$(date)
#current_time=`echo ${current_time} | sed 's/\ /-/g' | sed 's/://g'` #${current_time// /_}
#mkdir -p res-$current_time
#cp run_dist_tf.sh res-$current_time
cd $DIST_TF_LAUNCHER_DIR 
./run_dist_tf.sh

# deactivate virtualenv
#deactivate
