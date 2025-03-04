# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
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
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import json
import os
import sys

import numpy as np
import tensorflow as tf
import graphlearn as gl
import graphlearn.python.nn.tf as tfg

from ego_rgcn import EgoRGCN

flags = tf.app.flags
FLAGS = flags.FLAGS
# user-defined params
flags.DEFINE_integer('epochs', 40, 'training epochs')
flags.DEFINE_integer('train_batch_size', 128, 'training minibatch size')
flags.DEFINE_integer('test_batch_size', 128, 'test minibatch size')
flags.DEFINE_float('learning_rate', 0.05, 'learning rate')
flags.DEFINE_float('drop_out', 0.5, 'drop out rate')
flags.DEFINE_integer('hidden_dim', 128, 'hidden layer dim')
flags.DEFINE_integer('class_num', 7, 'final output embedding dim')
flags.DEFINE_string('nbrs_num', '[20]', 'string of list, neighbor num of each hop')
flags.DEFINE_string('agg_type', 'mean', 'aggregation type, mean, max or sum')
flags.DEFINE_string('sampler', 'random', 'neighbor sampler strategy. random, in_degree, topk.')
flags.DEFINE_integer('num_relations', 2, 'number of relations')
flags.DEFINE_integer('num_bases', 1, 'number of bases to use for RGCNConv')
flags.DEFINE_integer('num_blocks', None, 'number of blocks to use for RGCNConv')
flags.DEFINE_string('attr_types', None, 'node attribute types')
flags.DEFINE_string('attr_dims', None, 'node attribute dimensions')
flags.DEFINE_integer('float_attr_num', 1433, 
  'number of float attrs. If there is only float attrs, we use this flag to instead of two above flags.')


if FLAGS.attr_types is not None and FLAGS.attr_dims is not None:
  attr_types = json.loads(FLAGS.attr_types)
  attr_dims = json.loads(FLAGS.attr_dims)
else:
  assert FLAGS.float_attr_num > 0
  attr_types = ['float'] * FLAGS.float_attr_num
  attr_dims = [0] * FLAGS.float_attr_num
nbrs_num = json.loads(FLAGS.nbrs_num)

def load_graph():
  """ Load node and edge data to build graph.
    Note that node_type must be "i", and edge_type must be "r_i", 
    the number of edge tables must be the same as FLAGS.num_relations.
  """
  cur_path = sys.path[0]
  dataset_folder = os.path.join(cur_path, '../../data/cora/')
  g = gl.Graph()\
        .node(dataset_folder + "node_table", node_type="i",
              decoder=gl.Decoder(labeled=True,
                                 attr_types=attr_types,
                                 attr_delimiter=":"))                      \
        .edge(dataset_folder + "edge_table",
              edge_type=("i", "i", "r_0"),
              decoder=gl.Decoder(weighted=True), directed=False)           \
        .edge(dataset_folder + "edge_table_with_self_loop",
              edge_type=("i", "i", "r_1"),
              decoder=gl.Decoder(weighted=True), directed=False)           \
        .node(dataset_folder + "train_table", node_type="i",
              decoder=gl.Decoder(weighted=True), mask=gl.Mask.TRAIN)       \
        .node(dataset_folder + "test_table", node_type="i",
              decoder=gl.Decoder(weighted=True), mask=gl.Mask.TEST)
  return g

def query(graph, mask):
  """ k-hop neighbor sampling using different relations.
    For train, the query node name are as follows:
    root: ['train']
    1-hop neighbors: ['train_hop_0_r_0', 'train_hop_0_r_1']
    2-hop neighbors: ['train_hop_0_r_0_hop_1_r_0', 'train_hop_0_r_0_hop_1_r_0', 
                      'train_hop_0_r_1_hop_1_r_0', 'train_hop_0_r_1_hop_1_r_0']
    ...
  """
  prefix = ('train', 'test')[mask.value - 1]
  bs = FLAGS.train_batch_size if prefix == 'train' else FLAGS.test_batch_size
  q = graph.V("i", mask=mask).batch(bs).alias(prefix)
  current_hop_list = [q]
  for idx, hop in enumerate(nbrs_num):
    next_hop_list = []
    for hop_q in current_hop_list:
      for i in range(FLAGS.num_relations):
        alias = hop_q.get_alias() + '_hop_' + str(idx) + '_r_' + str(i)
        next_hop_list.append(hop_q.outV('r_'+str(i)).sample(hop).by(FLAGS.sampler).alias(alias))
    current_hop_list = next_hop_list
  return q.values()

def supervised_loss(logits, labels):
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  return tf.reduce_mean(loss)

def accuracy(logits, labels):
  indices = tf.math.argmax(logits, 1, output_type=tf.int32)
  correct = tf.reduce_sum(tf.cast(tf.math.equal(indices, labels), tf.float32))
  return correct / tf.cast(tf.shape(labels)[0], tf.float32)

def reformat_node_feature(data_dict, alias_list, feature_handler):
  """ Transforms and organizes the input data to a list of list,
  each element of list is also a list which consits of k-hop multi-relations
  neighbor nodes' feature tensor.
  """
  cursor = 0
  x = feature_handler.forward(data_dict[alias_list[cursor]])
  cursor += 1
  x_list = [[x]]
  
  nbr_list_len = FLAGS.num_relations
  for idx in range(len(nbrs_num)):
    nbr_list = []
    for i in range(nbr_list_len):
      nbr_list.append(feature_handler.forward(data_dict[alias_list[cursor]]))
      cursor += 1
    x_list.append(nbr_list)
    nbr_list_len *= FLAGS.num_relations
  return x_list

def train(graph, model):
  tfg.conf.training = True
  query_train = query(graph, gl.Mask.TRAIN)
  dataset = tfg.Dataset(query_train, window=5)
  data_dict = dataset.get_data_dict()
  feature_handler = tfg.FeatureHandler('feature_handler',
    query_train.get_node("train").decoder.feature_spec)

  x_list = reformat_node_feature(data_dict, query_train.list_alias(), feature_handler)
  train_embeddings = model.forward(x_list, nbrs_num)
  loss = supervised_loss(train_embeddings, data_dict['train'].labels)
  return dataset.iterator, loss

def test(graph, model):
  tfg.conf.training = False
  query_test = query(graph, gl.Mask.TEST)
  dataset = tfg.Dataset(query_test, window=5)
  data_dict = dataset.get_data_dict()
  feature_handler = tfg.FeatureHandler('feature_handler',
    query_test.get_node("test").decoder.feature_spec)

  x_list = reformat_node_feature(data_dict, query_test.list_alias(), feature_handler)
  test_embeddings = model.forward(x_list, nbrs_num)
  test_acc = accuracy(test_embeddings, data_dict['test'].labels)
  return dataset.iterator, test_acc, data_dict['test'].ids,\
    data_dict['test'].labels, tf.nn.softmax(test_embeddings)

def main(unused_argv):
  g = load_graph()
  g.init()
  # Define Model
  input_dim = sum([1 if not i else i for i in attr_dims])
  model = EgoRGCN(input_dim,
                  FLAGS.hidden_dim,
                  FLAGS.class_num,
                  len(nbrs_num),
                  FLAGS.num_relations,
                  FLAGS.num_bases,
                  FLAGS.num_blocks,
                  agg_type=FLAGS.agg_type,
                  dropout=FLAGS.drop_out)
  # train and test
  train_iterator, loss = train(g, model)
  optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
  train_op = optimizer.minimize(loss)
  train_ops = [loss, train_op]
  test_iterator, test_acc, _, _, _ = test(g, model)
  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.initializer)
    step = 0
    print("Start Training...")
    for i in range(FLAGS.epochs):
      try:
        while True:
          ret = sess.run(train_ops)
          print("Epoch {}, Iter {}, Loss {:.5f}".format(i, step, ret[0]))
          step += 1
      except tf.errors.OutOfRangeError:
        sess.run(train_iterator.initializer) # reinitialize dataset.

    print("Start Testing...")
    total_test_acc = []
    sess.run(test_iterator.initializer)
    try:
      while True:
        ret = sess.run(test_acc)
        total_test_acc.append(ret)
    except tf.errors.OutOfRangeError:
      print("Finished.")
    print('Test Accuracy is: {:.4f}'.format(np.mean(total_test_acc)))
  g.close()


if __name__ == "__main__":
  tf.app.run()