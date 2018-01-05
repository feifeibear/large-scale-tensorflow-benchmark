#!/bin/bash -l
#SBATCH -p debug
#SBATCH -N 1 
#SBATCH -C knl,quad,cache
#SBATCH -t 00:12:00
#SBATCH -L SCRATCH
#SBATCH -J google_benchmark 
#SBATCH --output=local_benchmark.%j.log

module load tensorflow/intel-head
export OMP_NUM_THREADS=66
KMP_AFFINITY="granularity=fine,noverbose,compact,1,0"
KMP_SETTINGS=1
KMP_BLOCKTIME=1

# load virtualenv
#export WORKON_HOME=~/Envs
#source $WORKON_HOME/tf-local/bin/activate

# train inception
SCRIPT_DIR=/global/cscratch1/sd/yyang420/fjr/tensorflow/distributed-tensorflow-benchmarks/google-benchmarks
cd $SCRIPT_DIR
srun -n 1 -N 1 -c 272 python tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--num_gpus=1 \
--batch_size=64 \
--num_warmup_batches=2 \
--num_batches=10 \
--data_format=NCHW \
--variable_update=parameter_server \
--local_parameter_device=cpu \
--device=cpu \
--optimizer=sgd \
--model=inception3 \
--data_name=imagenet
#--data_dir=/home/ubuntu/imagenet/

# deactivate virtualenv
#deactivate

