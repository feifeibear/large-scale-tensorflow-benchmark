module use /apps/daint/UES/6.0.UP02/sandbox-dl/modules/all
module load daint-gpu
module load TensorFlow/1.3.0-CrayGNU-17.08-cuda-8.0-python3

python3 imagenet_main.py --data_dir=/scratch/snx3000/youyang9/data/imagenet
