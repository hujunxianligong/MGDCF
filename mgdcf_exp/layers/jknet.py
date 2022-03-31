# coding = utf8

import os
import sys
import logging
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
from time import time
from grecx.evaluation.ranking import evaluate_mean_global_metrics
import grecx as grx
from grecx.datasets.light_gcn_dataset import LightGCNDataset
import tf_geometric as tfg
from grecx.layers import LightGCN
from tf_geometric import SparseAdj
from tf_geometric.nn import gcn_build_cache_for_graph, gcn_norm_adj
from tf_geometric.utils import tf_utils
import pickle
import scipy.sparse as sp
import tf_sparse as tfs
import warnings


class HeteroGCN(tf.keras.Model):
    """
    Each NGCF Convolutional Layer
    """

    def __init__(self, dense_activation=tf.nn.leaky_relu, edge_drop_rate=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dense_activation = dense_activation
        self.gcn_dense = None
        self.interaction_dense = None
        self.edge_drop_rate = edge_drop_rate

    def build(self, input_shape):
        x_shape, _ = input_shape
        self.gcn_dense = tf.keras.layers.Dense(x_shape[1], activation=self.dense_activation)

    @classmethod
    def build_virtual_edge_index(cls, user_item_edge_index, num_users=None):
        return LightGCN.build_virtual_edge_index(user_item_edge_index, num_users)

    def build_cache_for_graph(self, graph, override=False):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.
        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None
        """
        if override:
            graph.cache[LightGCN.CACHE_KEY] = None
        LightGCN.norm_adj(graph.edge_index, graph.num_nodes, cache=graph.cache)

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index = inputs[0], inputs[1]

        num_nodes = tfs.shape(x)[0]
        normed_adj = LightGCN.norm_adj(edge_index, num_nodes=num_nodes, cache=cache) \
            .dropout(self.edge_drop_rate, training=training)

        h = normed_adj @ x
        h = self.gcn_dense(h)

        return h


class HeteroJKNet(tf.keras.Model):

    def __init__(self, k=4, z_drop_rate=0.0, edge_drop_rate=0.0, dense_activation=tf.nn.leaky_relu, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hetero_gcns = [HeteroGCN(dense_activation=dense_activation, edge_drop_rate=edge_drop_rate) for _ in range(k)]
        self.dropout = tf.keras.layers.Dropout(z_drop_rate)

        for i, hetero_gcn in enumerate(self.hetero_gcns):
            setattr(self, "hetero_gcn{}".format(i), hetero_gcn)

    @classmethod
    def build_virtual_edge_index(cls, user_item_edge_index, num_users=None):
        return LightGCN.build_virtual_edge_index(user_item_edge_index, num_users)

    def build_cache_for_graph(self, graph, override=False):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.
        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None
        """
        self.hetero_gcns[0].build_cache_for_graph(graph, override=override)

    def call(self, inputs, training=None, mask=None, cache=None):
        x, edge_index = inputs

        h = x
        # h = self.dropout(h, training=training)
        h_list = [h]

        for hetero_gcn in self.hetero_gcns:
            h = hetero_gcn([h, edge_index], training=training, cache=cache)
            h = self.dropout(h, training=training)
            h_list.append(h)

        # h_list = [tf.nn.l2_normalize(h, axis=-1) for h in h_list]
        # h = tf.concat([x] + h_list, axis=-1)
        h = tf.reduce_mean(tf.stack(h_list, axis=0), axis=0)

        return h


