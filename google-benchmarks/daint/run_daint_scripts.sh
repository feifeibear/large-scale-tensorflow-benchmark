#!/bin/bash

OUTPUT_DIR=outputs
mkdir -p $OUTPUT_DIR

# ./run_dist_benchmarks_daint.sh TF_NUM_PS TF_NUM_WORKER parameter_server/distributed_replicated true/false

./run_dist_benchmarks_daint.sh 8 8 \
         parameter_server false resnet50 > \
         $OUTPUT_DIR/32_128_ps_false.out &

#for TF_NUM_WORKER in 64 ; do
#  #for TF_NUM_PS in {}; do
#    for VARIABLE_UPDATE in parameter_server distributed_replicated; do
#      for REAL_DATA in false; do
#        for TF_NUM_PS in `expr $TF_NUM_WORKER / 2` `expr $TF_NUM_WORKER / 4` `expr $TF_NUM_WORKER / 8`; do
#          echo -e "\nRunning $TF_NUM_PS $TF_NUM_WORKER $VARIABLE_UPDATE $REAL_DATA"
#          ./run_dist_benchmarks_daint.sh $TF_NUM_PS $TF_NUM_WORKER \
#           $VARIABLE_UPDATE $REAL_DATA > \
#           $OUTPUT_DIR/$TF_NUM_PS\_$TF_NUM_WORKER\_$VARIABLE_UPDATE\_$REAL_DATA.out
#        done
#      done
#    done
#  #done
#done
#
#for TF_NUM_WORKER in 512; do
#  for TF_NUM_PS in 128; do

#for TF_NUM_WORKER in 8; do
#  for TF_NUM_PS in 6; do
#    for VARIABLE_UPDATE in parameter_server; do
#      for REAL_DATA in false; do
#        echo -e "\nRunning $TF_NUM_PS $TF_NUM_WORKER $VARIABLE_UPDATE $REAL_DATA"
#        ./run_dist_benchmarks_cori.sh $TF_NUM_PS $TF_NUM_WORKER \
#         $VARIABLE_UPDATE $REAL_DATA > \
#         $OUTPUT_DIR/$TF_NUM_PS\_$TF_NUM_WORKER\_$VARIABLE_UPDATE\_$REAL_DATA.out
#      done
#    done
#  done
#done

#for TF_NUM_WORKER in 256; do
#  for TF_NUM_PS in 80; do
#    for VARIABLE_UPDATE in parameter_server; do
#      for REAL_DATA in false; do
#        echo -e "\nRunning $TF_NUM_PS $TF_NUM_WORKER $VARIABLE_UPDATE $REAL_DATA"
#        ./run_dist_benchmarks_cori.sh $TF_NUM_PS $TF_NUM_WORKER \
#         $VARIABLE_UPDATE $REAL_DATA > \
#         $OUTPUT_DIR/$TF_NUM_PS\_$TF_NUM_WORKER\_$VARIABLE_UPDATE\_$REAL_DATA.out
#      done
#    done
#  done
#done
#
#for TF_NUM_WORKER in 512; do
#  for TF_NUM_PS in 128; do
#    for VARIABLE_UPDATE in parameter_server; do
#      for REAL_DATA in false; do
#        echo -e "\nRunning $TF_NUM_PS $TF_NUM_WORKER $VARIABLE_UPDATE $REAL_DATA"
#        ./run_dist_benchmarks_cori.sh $TF_NUM_PS $TF_NUM_WORKER \
#         $VARIABLE_UPDATE $REAL_DATA > \
#         $OUTPUT_DIR/$TF_NUM_PS\_$TF_NUM_WORKER\_$VARIABLE_UPDATE\_$REAL_DATA.out
#      done
#    done
#  done
#done
#
