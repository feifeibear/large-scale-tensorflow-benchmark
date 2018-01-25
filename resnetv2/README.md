<font size=4><b>Distributed ResNet on Cifar and Imagenet Dataset.</b></font>
This Repo contains the code for Distributed ResNet Training.
contact: Jiarui Fang (fjr14@mails.tsinghua.edu.cn)
I met the same problem with SyncReplicaOptimzor as mentioned in
[github issue](https://github.com/tensorflow/tensorflow/issues/6976)
[tensorflow](https://stackoverflow.com/questions/42006967/scalability-issues-related-to-distributed-tensorflow)


<b>Cifar-10 Settings:</b>

* Random split 50k training set into 45k/5k train/eval split.
* Pad to 36x36 and random crop. Horizontal flip. Per-image whitening.
* Momentum optimizer 0.9.
* Learning rate schedule: 0.1 (40k), 0.01 (60k), 0.001 (>60k).
* L2 weight decay: 0.0002.
* Batch size: 128.

<b>Results with this code:</b>
1. Cifar-10
global batch size = 128, eval results are as following.

CIFAR-10 Model|Best Precision|PS-WK |Steps|Speed (stp/sec)
--------------|--------------|------|-----|--------------
50 layer|93.6%|local|~80k|13.94
50 layer|85.2%|1ps-1wk|~80k|10.19
50 layer|86.4%|2ps-4wk|~80k|20.3
50 layer|87.3%|4ps-8wk|~60k|19.19

Distributed Versions get lower eval accuracy results as provided in [Tensorflow Model Research](https://github.com/tensorflow/models/tree/master/research/resnet)

2. ImageNet
We set global batch size as 128\*8 = 1024.
Follows the Hyperparameter settting in [Intel-Caffe](https://github.com/intel/caffe/tree/master/models/intel_optimized_models/multinode/resnet_50_8_nodes)
ImageNet Model|Best Precision|PS-WK |Steps|Speed (stp/sec)
--------------|--------------|------|-----|--------------
50 layer|62.6%| 8-ps-8wk| ~76k | 0.93
50 layer|64.4%| 4-ps-8wk| ~75k | 0.90


<b>Prerequisite:</b>

1. Install TensorFlow, Bazel.
Piz Daint dose not provide a Bazel combining with its default TensorFlow. Instead, I install a conda2 package on Daint. Bazel and other packages required, such as an CPU tensorflow, are installed by virtualenv inside conda2.

2. Download ImageNet Dataset to Daint
To avoid the error raised from unrecognition of the relative directory path, the following modification should made in download_and_preprocess_imagenet.sh.
# old:
WORK_DIR="$0.runfiles/inception/inception"
# new:
WORK_DIR="$(realpath -s "$0").runfiles/inception/inception"
After few days, you will see the following data in your data path.
Due to the file system of Daint dose not support storage of millions of files, I deleted raw-data directory.


3. Download CIFAR-10/CIFAR-100 dataset.
```shell
curl -o cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
curl -o cifar-100-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
```

<b>How to run:</b>

```shell
# cd to the models repository and run with bash. Expected command output shown.
# The directory should contain an empty WORKSPACE file, the resnet code, and the cifar10 dataset.
# Note: The user can split 5k from train set for eval set.
$ ls -R
.:
cifar10  resnet  WORKSPACE

./cifar10:
data_batch_1.bin  data_batch_2.bin  data_batch_3.bin  data_batch_4.bin
data_batch_5.bin  test_batch.bin

$ cd tools
# run local for cifar10. It will launch 1 ps and 2 workers
$ sh submit_local_dist.sh
# run on Piz Daint for cifar
$ sh submit_cifar_daint_dist.sh #server #worker #batch_size
# run on Piz Daint for Imagenet
$ sh submit_imagenet_daint_dist.sh #server #worker
```

<b>Related papers:</b>

Identity Mappings in Deep Residual Networks

https://arxiv.org/pdf/1603.05027v2.pdf

Deep Residual Learning for Image Recognition

https://arxiv.org/pdf/1512.03385v1.pdf

Wide Residual Networks

https://arxiv.org/pdf/1605.07146v1.pdf


