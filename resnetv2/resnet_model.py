# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six
import resnet_model_official

from tensorflow.python.training import moving_averages

FLAGS = tf.app.flags.FLAGS


HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')


class ResNet(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.train.get_or_create_global_step()
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()
    self.summaries = tf.summary.merge_all()

  def _build_model(self):
    """Build the core model within the graph."""
    if FLAGS.dataset == 'cifar10':
      network = resnet_model_official.cifar10_resnet_v2_generator(resnet_size= 50, num_classes = 10, data_format=None)
    elif FLAGS.dataset == 'imagenet':
      network = resnet_model_official.imagenet_resnet_v2(resnet_size = 50, num_classes = 1001, data_format=None)

    logits = network(self._images, True)
    self.predictions = tf.nn.softmax(logits)
    cross_entropy = tf.losses.softmax_cross_entropy(
	      logits=logits, onehot_labels=self.labels
	)
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss.
    self.cost = cross_entropy +self.hps.weight_decay_rate * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    tf.summary.scalar('cost', self.cost)

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.summary.scalar('learning_rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    if self.hps.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)

    #dist: add sync 
    if FLAGS.sync_replicas:
      if FLAGS.replicas_to_aggregate is None:
        raise ValueError("Must specify an explicit `replicas_to_aggregate`")
      else:
        replicas_to_aggregate = FLAGS.replicas_to_aggregate

      optimizer = tf.train.SyncReplicasOptimizer(
          optimizer,
          replicas_to_aggregate=replicas_to_aggregate,
          total_num_replicas=FLAGS.replicas_to_aggregate,
          name="resnet_sync_replicas")

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')
    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

# dist sync init

    if FLAGS.sync_replicas:
      is_chief = (FLAGS.task_index == 0)
      self.replicas_hook = optimizer.make_session_run_hook(is_chief)
      self.local_init_op = optimizer.local_step_init_op
      if is_chief:
        self.local_init_op = optimizer.chief_init_op

      self.ready_for_local_init_op = optimizer.ready_for_local_init_op

      # Initial token and chief queue runners required by the sync_replicas mode
      self.chief_queue_runner = optimizer.get_chief_queue_runner()
      self.sync_init_op = optimizer.get_init_tokens_op()

    self.init_op = tf.global_variables_initializer()
