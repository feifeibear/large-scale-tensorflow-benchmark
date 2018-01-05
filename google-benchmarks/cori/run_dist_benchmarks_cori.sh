#!/bin/bash
set -x

# Arguments:
#   $1: TF_NUM_PS: number of parameter servers
#   $2: TF_NUM_WORKER: number of workers
#   $3: variable_update: parameter_server/distributed_replicated
#   $4: real_data: true/false
#   $5: num_executions: number of executions to run (default 5)

USERNAME=yyang420
SCRIPT_DIR=/global/cscratch1/sd/yyang420/fjr/tensorflow/distributed-tensorflow-benchmarks/google-benchmarks/cori
TF_DIR=$SCRATCH/fjr/tf/tf-$1-$2-$3-$4-$5 #?
SCRIPT_NAME=google-benchmarks_dist_cori.sh
SCRIPT_ARGS="$1 $2 $3 $4 $5"

# get number of executions
#  if [ -z "$5" ]; then
#    # default to 5
#    num_executions=1
#  else
#    num_executions=$5
#  fi
#  
#  num_nodes=$2

mkdir -p $TF_DIR
cp ./run_dist_tf.sh $TF_DIR

sbatch -N $2 $SCRIPT_DIR/$SCRIPT_NAME $SCRIPT_ARGS

#sleep 1800
#exe_sum=0
#for file in `ls $TF_DIR/worker*`; do
#  node_result=`tail -2 $file | head -n 1 | cut -d' ' -f3`
#  exe_sum=`echo $exe_sum $node_result | awk '{print $1 + $2}'`
#done
##rm $TF_DIR/ps*
##rm $TF_DIR/worker*
#exe_result=`echo $exe_sum $num_nodes | awk '{print $1/$2}'`
#echo "exe_result: $exe_result"
