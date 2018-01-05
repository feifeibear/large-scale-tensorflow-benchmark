#!/bin/bash

# Arguments:
#   $1: TF_NUM_PS: number of parameter servers
#   $2: TF_NUM_WORKER: number of workers
#   $3: variable_update: parameter_server/distributed_replicated
#   $4: real_data: true/false
#   $5: num_executions: number of executions to run (default 5)

USERNAME=youyang9
SCRIPT_DIR=`pwd`
TF_DIR=$SCRATCH/fjr/tf/tf-$1-$2-$3-$4-$5 #?
SCRIPT_NAME=google-benchmarks_dist_daint.sh
SCRIPT_ARGS="$1 $2 $3 $4 $5"

mkdir -p $TF_DIR
cp ./run_dist_tf.sh $TF_DIR

sbatch -N $2 $SCRIPT_DIR/$SCRIPT_NAME $SCRIPT_ARGS
