#!/bin/bash

# load modules
module use /apps/daint/UES/6.0.UP02/sandbox-dl/modules/all
module load daint-gpu
module load TensorFlow/1.2.1-CrayGNU-17.08-cuda-8.0-python3

# create virtualenv
export WORKON_HOME=~/Envs
mkdir -p $WORKON_HOME
cd $WORKON_HOME
virtualenv tf-daint

# load virtualenv
source $WORKON_HOME/tf-daint/bin/activate

# install dependencies
pip install -r /scratch/snx3000/youyang9/fjr/tf_workspace/large-scale-tensorflow-benchmark/environments_setup/requirements.txt
pip install absl-py

# install aws-cli
pip install awscli

# deactivate virtualenv
deactivate
