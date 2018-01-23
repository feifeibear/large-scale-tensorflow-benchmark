# load modules
module use /apps/daint/UES/6.0.UP02/sandbox-dl/modules/all
module load daint-gpu
# module load TensorFlow/1.2.1-CrayGNU-17.08-cuda-8.0-python3
module load TensorFlow/1.3.0-CrayGNU-17.08-cuda-8.0-python3



#python3 ./cifar10_main.py --data_dir=. --model_dir=./model
python3 ./resnet_cifar_eval.py --data_dir=$SCRATCH/data --log_root=./model --eval_dir=./model/test --eval_once=True
