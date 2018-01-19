#!/bin/bash

#SBATCH --job-name=resnet
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --constraint=gpu
#SBATCH --output=resnet.%j.log

# load modules
module use /apps/daint/UES/6.0.UP02/sandbox-dl/modules/all
module load daint-gpu
module load TensorFlow/1.2.1-CrayGNU-17.08-cuda-8.0-python3

SLURM_WORKER_HOSTS=$(scontrol show hostnames ${SLURM_NODELIST} | 
                       head -n 1 | tr -s '\n' ',' | 
                       head --bytes -1)

SLURM_EVAL_HOSTS=$(scontrol show hostnames ${SLURM_NODELIST} | 
                       tail -n 1 | tr -s '\n' ',' | 
                       head --bytes -1)

echo "worker is $SLURM_WORKER_HOSTS"
echo "evaler is $SLURM_EVAL_HOSTS"

rm -rf ./tmp
mkdir -p ./tmp


srun --no-kill --nodelist ${SLURM_WORKER_HOSTS} -n 1 -N 1 python3 resnet_main.py \
                               --train_data_path=./cifar-10-batches-bin/data_batch* \
                               --log_root=./tmp/resnet_model \
                               --train_dir=./tmp/resnet_model/train \
                               --dataset='cifar10' \
                               --num_gpus=1 &> train.${SLURM_JOBID}.log &



srun --no-kill --nodelist ${SLURM_EVAL_HOSTS} -n 1 -N 1 python3 resnet_main.py \
                               --eval_data_path=./cifar-10-batches-bin/test_batch* \
                               --log_root=./tmp/resnet_model \
                               --eval_dir=./tmp/resnet_model/test \
                               --mode=eval \
                               --dataset='cifar10' \
                               --num_gpus=1 &> test.${SLURM_JOBID}.log



