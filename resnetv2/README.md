<font size=4><b>Distributed ResNet on Cifar and Imagenet Dataset.</b></font>

contact: Jiarui Fang (fjr14@mails.tsinghua.edu.cn)

<b>Dataset:</b>

https://www.cs.toronto.edu/~kriz/cifar.html

<b>Related papers:</b>

Identity Mappings in Deep Residual Networks

https://arxiv.org/pdf/1603.05027v2.pdf

Deep Residual Learning for Image Recognition

https://arxiv.org/pdf/1512.03385v1.pdf

Wide Residual Networks

https://arxiv.org/pdf/1605.07146v1.pdf

<b>Cifar-10 Settings:</b>

* Random split 50k training set into 45k/5k train/eval split.
* Pad to 36x36 and random crop. Horizontal flip. Per-image whitening.
* Momentum optimizer 0.9.
* Learning rate schedule: 0.1 (40k), 0.01 (60k), 0.001 (>60k).
* L2 weight decay: 0.0002.
* Batch size: 128. (28-10 wide and 1001 layer bottleneck use 64)

<b>Results with this code:</b>
global batch size = 128

CIFAR-10 Model|Best Precision|PS-WK |Steps|Speed (stp/sec)
--------------|--------------|------|-----|--------------
50 layer|93.6%|local|~80k|13.94
50 layer|85.2%|1ps-1wk|~80k|10.19
50 layer|86.4%|2ps-4wk|~80k|20.3
50 layer|87.3%|4ps-8wk|~60k|19.19

I met the same problem with SyncReplicaOptimzor as mentioned in
[github issue](https://github.com/tensorflow/tensorflow/issues/6976)
[tensorflow](https://stackoverflow.com/questions/42006967/scalability-issues-related-to-distributed-tensorflow)


<b>Results in Tensorflow Model Research:</b>

CIFAR-10 Model|Best Precision|Steps
--------------|--------------|------
32 layer|92.5%|~80k
110 layer|93.6%|~80k
164 layer bottleneck|94.5%|~80k
1001 layer bottleneck|94.9%|~80k
28-10 wide|95%|~90k

CIFAR-100 Model|Best Precision|Steps
---------------|--------------|-----
32 layer|68.1%|~45k
110 layer|71.3%|~60k
164 layer bottleneck|75.7%|~50k
1001 layer bottleneck|78.2%|~70k
28-10 wide|78.3%|~70k

<b>Prerequisite:</b>

1. Install TensorFlow, Bazel.

2. Download CIFAR-10/CIFAR-100 dataset.

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
$ sh submit_cifar_daint_dist.sh #server #worker
# run on Piz Daint for Imagenet
$ sh submit_imagenet_daint_dist.sh #server #worker
```
