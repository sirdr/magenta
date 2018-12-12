# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""The evaluation script.

This script requires tensorflow 1.1.0-rc1 or beyond.
As of 04/05/17 this requires installing tensorflow from source,
(https://github.com/tensorflow/tensorflow/releases)

So that it works locally, the default worker_replicas and total_batch_size are
set to 1. For training in 200k iterations, they both should be 32.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import pickle

from magenta.models.nsynth import utils

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("master", "",
                           "BNS name of the TensorFlow master to use.")
tf.app.flags.DEFINE_string("config", "h512_bo16", "Model configuration name")
tf.app.flags.DEFINE_integer("task", 0,
                            "Task id of the replica running the training.")
tf.app.flags.DEFINE_integer("worker_replicas", 1,
                            "Number of replicas. We train with 32.")
tf.app.flags.DEFINE_integer("ps_tasks", 0,
                            "Number of tasks in the ps job. If 0 no ps job is "
                            "used. We typically use 11.")
tf.app.flags.DEFINE_integer("total_batch_size", 1,
                            "Batch size spread across all sync replicas."
                            "We use a size of 32.")
tf.app.flags.DEFINE_integer("sample_length", 64000,
                            "Raw sample length of input.")
tf.app.flags.DEFINE_integer("num_evals", None,
                            "number of evauaitons -- None does entire dataset")
tf.app.flags.DEFINE_string("logdir", "/tmp/nsynth",
                           "The log directory for this experiment.")
tf.app.flags.DEFINE_string("checkpoint_dir", "/tmp/nsynth",
                           "Where the checkpoints are stored")
tf.app.flags.DEFINE_string("checkpoint_path", None,
                           "path of checkpoint -- if none use checkpoint_dir")
tf.app.flags.DEFINE_string("problem", "nsynth",
                           "Which problem setup (i.e. dataset) to use")
tf.app.flags.DEFINE_string("eval_path", "", "The path to the train tfrecord.")
tf.app.flags.DEFINE_string("log", "INFO",
                           "The threshold for what messages will be logged."
                           "DEBUG, INFO, WARN, ERROR, or FATAL.")
tf.app.flags.DEFINE_bool("vae", False,
                           "Whether or not to train variationally")
tf.app.flags.DEFINE_bool("small", False,
                           "Whether to use full model i.e. 30 layers in decoder/encoder or reduced model")
tf.app.flags.DEFINE_bool("asymmetric", False,
                           "Whether to have equal number of layers in decoder/encoder or a weaker decoder")
tf.app.flags.DEFINE_bool("kl_annealing", False,
                           "Whether to use kl_annealing")
tf.app.flags.DEFINE_float("aux_coefficient", 0,
                           "coefficient for auxilliary loss")
tf.app.flags.DEFINE_float("annealing_loc", 1750.,
                           "params of normal cdf for annealing")
tf.app.flags.DEFINE_float("annealing_scale", 150.,
                           "params of normal cdf for annealing")
tf.app.flags.DEFINE_float("kl_threshold", None,
                           "Threshold with which to bound KL-Loss")
tf.app.flags.DEFINE_float("input_dropout", 1,
                           "How much dropout at input to add")

def main(unused_argv=None):
  tf.logging.set_verbosity(FLAGS.log)

  if FLAGS.config is None:
    raise RuntimeError("No config name specified.")

  if FLAGS.vae:
    config = utils.get_module("wavenet." + FLAGS.config).VAEConfig(
        FLAGS.eval_path, sample_length=FLAGS.sample_length, problem=FLAGS.problem, small=FLAGS.small, asymmetric=FLAGS.asymmetric, aux=FLAGS.aux_coefficient, dropout=FLAGS.input_dropout)
  else:
    config = utils.get_module("wavenet." + FLAGS.config).Config(
        FLAGS.eval_path, sample_length=FLAGS.sample_length, problem=FLAGS.problem, small=FLAGS.small, asymmetric=FLAGS.asymmetric)

  logdir = FLAGS.logdir
  tf.logging.info("Saving to %s" % logdir)

  with tf.Graph().as_default():
    total_batch_size = FLAGS.total_batch_size
    assert total_batch_size % FLAGS.worker_replicas == 0
    worker_batch_size = total_batch_size / FLAGS.worker_replicas

    # Run the Reader on the CPU
    cpu_device = "/job:localhost/replica:0/task:0/cpu:0"
    if FLAGS.ps_tasks:
      cpu_device = "/job:worker/cpu:0"

    with tf.device(cpu_device):
      inputs_dict = config.get_batch(worker_batch_size, is_training=False)

    with tf.device(
        tf.train.replica_device_setter(ps_tasks=FLAGS.ps_tasks,
                                       merge_devices=True)):
      global_step = tf.get_variable(
          "global_step", [],
          tf.int32,
          initializer=tf.constant_initializer(0),
          trainable=False)

      # build the model graph
      outputs_dict = config.build(inputs_dict, is_training=False)

      if FLAGS.vae:
        if FLAGS.kl_annealing:
          dist = tfp.distributions.Normal(loc=FLAGS.annealing_loc, scale=FLAGS.annealing_scale)
          annealing_rate = dist.cdf(tf.to_float(global_step)) # how to adjust the annealing
        else:
          annealing_rate = 0.
        kl = outputs_dict["loss"]["kl"]
        rec = outputs_dict["loss"]["rec"]
        aux = outputs_dict["loss"]["aux"]
        tf.summary.scalar("kl", kl)
        tf.summary.scalar("rec", rec)
        tf.summary.scalar("annealing_rate", annealing_rate)
        if FLAGS.kl_threshold is not None:
          kl = tf.maximum(tf.cast(FLAGS.kl_threshold, dtype=kl.dtype), kl)
        if FLAGS.aux_coefficient > 0:
          tf.summary.scalar("aux", aux)
        loss = rec + annealing_rate*kl + tf.cast(FLAGS.aux_coefficient, dtype=tf.float32)*aux
      else:
        loss = outputs_dict["loss"]
        
      tf.summary.scalar("train_loss", loss)

      labels = inputs_dict["parameters"]
      x_in = inputs_dict["wav"]
      batch_size, _ = x_in.get_shape().as_list()
      predictions = outputs_dict["predictions"]
      _, pred_dim = predictions.get_shape().as_list()
      predictions = tf.reshape(predictions, [batch_size, -1, pred_dim])
      encodings = outputs_dict["encoding"]


      session_config = tf.ConfigProto(allow_soft_placement=True)

      # Define the metrics:
      if FLAGS.vae:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'eval/kl': slim.metrics.streaming_mean(kl),
            'eval/rec': slim.metrics.streaming_mean(rec),
            'eval/loss': slim.metrics.streaming_mean(loss),
            'eval/predictions': slim.metrics.streaming_concat(predictions),
            'eval/labels': slim.metrics.streaming_concat(labels),
            'eval/encodings': slim.metrics.streaming_concat(encodings),
            'eval/audio': slim.metrics.streaming_concat(x_in)
        })
      else:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'eval/loss': slim.metrics.streaming_mean(loss),
            'eval/predictions': slim.metrics.streaming_concat(predictions),
            'eval/labels': slim.metrics.streaming_concat(labels),
            'eval/encodings': slim.metrics.streaming_concat(encodings),
            'eval/audio': slim.metrics.streaming_concat(x_in)
        })

      print('Running evaluation Loop...')
      if FLAGS.checkpoint_path is not None:
        checkpoint_path = FLAGS.checkpoint_path
      else:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
      metric_values = slim.evaluation.evaluate_once(
          num_evals=FLAGS.num_evals,
          master=FLAGS.master,
          checkpoint_path=checkpoint_path,
          logdir=FLAGS.logdir,
          eval_op=names_to_updates.values(),
          final_op=names_to_values.values(),
          session_config=session_config)

      names_to_values = dict(zip(names_to_values.keys(), metric_values))

      losses = {}
      for k, v in names_to_values.items():
        name = k.split('/')[-1]
        if name in ['predictions', 'encodings', 'labels', 'audio']:
          outpath = os.path.join(FLAGS.logdir, name)
          if name == 'predictions':
            v = np.argmax(v, axis = -1)
            v = utils.inv_mu_law_numpy(v - 128)
          np.save(outpath, v)
        else:
          losses[name] = v

      outpath_loss = os.path.join(FLAGS.logdir, 'losses.pickle')
      with open(outpath_loss, 'w') as w:
        pickle.dump(losses, w)




def console_entry_point():
  tf.app.run(main)


if __name__ == "__main__":
  console_entry_point()