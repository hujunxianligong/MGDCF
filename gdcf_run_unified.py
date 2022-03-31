# coding = utf8

import os
import sys
from argparse import ArgumentParser
import logging
from collections import Counter

from grecx.layers import LightGCN
import json
import tensorflow as tf
import numpy as np
from time import time
from grecx.evaluation.ranking import evaluate_mean_global_metrics
import grecx as grx
from grecx.datasets.light_gcn_dataset import LightGCNDataset
import tf_geometric as tfg
from tf_geometric import SparseAdj
from tf_geometric.nn import gcn_build_cache_for_graph, gcn_norm_adj
from tf_geometric.utils import tf_utils
import pickle
import scipy.sparse as sp
import tf_sparse as tfs
import warnings
# tf.config.run_functions_eagerly(True)
from mgdcf_exp.layers.jknet import HeteroJKNet
from mgdcf_exp.layers.mgdcf import build_homo_adjs, HomoMGDN, HeteroMGDN
from mgdcf_exp.utils import read_cache
import time

np.set_printoptions(precision=4)

logging.basicConfig(format='%(asctime)s %(message)s\t', level=logging.INFO, stream=sys.stdout)

parser = ArgumentParser()
parser.add_argument("method", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--gpu_ids", type=str, required=True)
parser.add_argument("--emb_size", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--lr_decay", type=float, required=True)
parser.add_argument("--z_l2_coef", type=float, required=True)
parser.add_argument("--num_negs", type=int, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--num_epochs", type=int, required=True)
parser.add_argument("--output_dir", type=str, required=True)

parser.add_argument("--adj_drop_rate", type=float, required=False)
parser.add_argument("--alpha", type=float, required=False)
parser.add_argument("--beta", type=float, required=False)
parser.add_argument("--num_iter", type=int, required=False)
parser.add_argument("--x_drop_rate", type=float, required=False)
parser.add_argument("--z_drop_rate", type=float, required=False)
parser.add_argument("--edge_drop_rate", type=float, required=False)

args = parser.parse_args()
logging.info(args)

method = args.method
dataset = args.dataset
gpu_ids = args.gpu_ids
embedding_size = args.emb_size
lr = args.lr
lr_decay = args.lr_decay
z_l2_coef = args.z_l2_coef
num_negs = args.num_negs
batch_size = args.batch_size
num_epochs = args.num_epochs
output_dir = args.output_dir

adj_drop_rate = args.adj_drop_rate
alpha = args.alpha
beta = args.beta
num_iter = args.num_iter

x_drop_rate = args.x_drop_rate
z_drop_rate = args.z_drop_rate
edge_drop_rate = args.edge_drop_rate
# k_l2_coef = args.k_l2_coef
# output_dir = args.output_dir

cache_base_dir = "cache"

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


# gpu_devices = tf.config.list_physical_devices("GPU")
# for gpu_device in gpu_devices:
#     tf.config.set_logical_device_configuration(gpu_device,
#                                                [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 20)])


class CFTask(object):

    def __init__(self):
        pass

    def load_data(self):
        # dataset = "light_gcn_yelp"  # "light_gcn_yelp" | "light_gcn_gowalla" | "light_gcn_amazon-book"

        data_dict = LightGCNDataset(dataset).load_data()

        self.num_users = data_dict["num_users"]
        self.num_items = data_dict["num_items"]
        self.user_item_edges = data_dict["user_item_edges"]
        self.train_index = data_dict["train_index"]
        self.train_user_items_dict = data_dict["train_user_items_dict"]
        self.test_user_items_dict = data_dict["test_user_items_dict"]

        self.train_user_item_edges = self.user_item_edges[self.train_index]
        self.train_user_item_edge_index = self.train_user_item_edges.transpose()

    def build_homo_adjs(self):
        train_user_item_edges = np.array(self.train_user_item_edges)

        adj_cache_path = os.path.join(cache_base_dir, "{}_adj_{}.p".format(dataset, adj_drop_rate))

        def build_adjs_func():
            return build_homo_adjs(train_user_item_edges, self.num_users, self.num_items, adj_drop_rate=adj_drop_rate)

        user_user_adj, item_item_adj = read_cache(adj_cache_path, func=build_adjs_func)

        return user_user_adj, item_item_adj

    def run(self):
        # result_dir = os.path.join("results", method)
        # os.makedirs(result_dir, exist_ok=True)
        # result_path = os.path.join(result_dir, "{}.json".format(dataset))
        #
        time_stamp = int(time.time() * 1000)

        dataset_output_dir = os.path.join(output_dir, dataset)
        os.makedirs(dataset_output_dir, exist_ok=True)
        result_path = os.path.join(dataset_output_dir,
                                   "{}_{}_sp_{}_{}.json".format(method, dataset, adj_drop_rate, time_stamp))

        with open(result_path, "a", encoding="utf-8") as f:
            f.write("{}\n".format(json.dumps(vars(args))))

        self.load_data()

        if method in ["MF"]:

            user_embeddings = tf.Variable(
                tf.random.truncated_normal([self.num_users, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
            item_embeddings = tf.Variable(
                tf.random.truncated_normal([self.num_items, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))

            z_dropout = tf.keras.layers.Dropout(z_drop_rate)

            @tf.function
            def forward(training=False):
                user_h = z_dropout(user_embeddings, training=training)
                item_h = z_dropout(item_embeddings, training=training)
                return user_h, item_h

        elif method.startswith("Hetero") or method in ["LightGCN", "APPNP", "JKNet", "DropEdge"]:
            virtual_graph = tfg.Graph(
                x=tf.Variable(
                    # initializer([int(num_users + num_items), int(embedding_size)]),
                    tf.random.truncated_normal([self.num_users + self.num_items, embedding_size],
                                               stddev=1 / np.sqrt(embedding_size)),
                    name="virtual_embeddings"
                ),
                edge_index=LightGCN.build_virtual_edge_index(self.train_user_item_edge_index, self.num_users)
            )

            if method in ["HeteroMGDCF", "LightGCN", "APPNP"]:
                model = HeteroMGDN(k=num_iter, alpha=alpha, beta=beta, edge_drop_rate=edge_drop_rate)
                model.build_cache_for_graph(virtual_graph)

                x_dropout = tf.keras.layers.Dropout(x_drop_rate)
                z_dropout = tf.keras.layers.Dropout(z_drop_rate)

                @tf_utils.function
                def forward(training=False):

                    virtual_h = x_dropout(virtual_graph.x, training=training)
                    virtual_h = model([virtual_h, virtual_graph.edge_index], training=training,
                                      cache=virtual_graph.cache)
                    virtual_h = z_dropout(virtual_h, training=training)

                    user_h = virtual_h[:self.num_users]
                    item_h = virtual_h[self.num_users:]
                    return user_h, item_h
            elif method in ["JKNet", "DropEdge"]:
                if method == "JKNet" and edge_drop_rate > 0.0:
                    raise Exception("JKNet edge_drop_rate")
                model = HeteroJKNet(k=num_iter, z_drop_rate=z_drop_rate, edge_drop_rate=edge_drop_rate)
                model.build_cache_for_graph(virtual_graph)

                x_dropout = tf.keras.layers.Dropout(x_drop_rate)

                # z_dropout = tf.keras.layers.Dropout(z_drop_rate)

                @tf_utils.function
                def forward(training=False):

                    virtual_h = x_dropout(virtual_graph.x, training=training)
                    virtual_h = model([virtual_h, virtual_graph.edge_index], training=training,
                                      cache=virtual_graph.cache)
                    # virtual_h = z_dropout(virtual_h, training=training)

                    user_h = virtual_h[:self.num_users]
                    item_h = virtual_h[self.num_users:]
                    return user_h, item_h

        elif method.startswith("Homo"):

            user_embeddings = tf.Variable(
                tf.random.truncated_normal([self.num_users, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
            item_embeddings = tf.Variable(
                tf.random.truncated_normal([self.num_items, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))

            user_user_adj, item_item_adj = self.build_homo_adjs()

            user_user_edge_index = np.stack([user_user_adj.row, user_user_adj.col], axis=0)
            user_user_edge_weight = user_user_adj.data
            item_item_edge_index = np.stack([item_item_adj.row, item_item_adj.col], axis=0)
            item_item_edge_weight = item_item_adj.data

            user_graph = tfg.Graph(user_embeddings, user_user_edge_index,
                                   edge_weight=user_user_edge_weight).convert_edge_to_directed(merge_mode="max")
            item_graph = tfg.Graph(item_embeddings, item_item_edge_index,
                                   edge_weight=item_item_edge_weight).convert_edge_to_directed(merge_mode="max")

            if method == "HomoMGDCF":
                # user_model = tfg.layers.APPNP([], k=0, dense_drop_rate=0.0, last_dense_drop_rate=0.0,
                #                               edge_drop_rate=edge_drop_rate, alpha=alpha)
                # item_model = tfg.layers.APPNP([], k=num_iter, dense_drop_rate=0.0, last_dense_drop_rate=0.0,
                #                               edge_drop_rate=edge_drop_rate, alpha=alpha)

                user_model = HomoMGDN(k=0, alpha=alpha, beta=beta, edge_drop_rate=edge_drop_rate)
                item_model = HomoMGDN(k=num_iter, alpha=alpha, beta=beta, edge_drop_rate=edge_drop_rate)

                x_dropout = tf.keras.layers.Dropout(x_drop_rate)
                z_dropout = tf.keras.layers.Dropout(z_drop_rate)

                user_model.build_cache_for_graph(user_graph)
                item_model.build_cache_for_graph(item_graph)

                @tf.function
                def forward(training=False):
                    user_h = x_dropout(user_graph.x, training=training)
                    item_h = x_dropout(item_graph.x, training=training)

                    user_h = user_model([user_h, user_graph.edge_index, user_graph.edge_weight],
                                        training=training, cache=user_graph.cache)
                    item_h = item_model([item_h, item_graph.edge_index, item_graph.edge_weight],
                                        training=training, cache=item_graph.cache)
                    user_h = z_dropout(user_h, training=training)
                    item_h = z_dropout(item_h, training=training)
                    return user_h, item_h
            else:
                raise Exception("wrong method name: {}".format(method))

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        @tf_utils.function
        def train_step(batch_user_indices, batch_item_indices):
            batch_neg_item_indices = tf.random.uniform(
                [tf.shape(batch_item_indices)[0], num_negs],
                0, self.num_items, dtype=tf.int32
            )

            with tf.GradientTape() as tape:
                user_h, item_h = forward(training=True)

                embedded_users = tf.gather(user_h, batch_user_indices)
                embedded_items = tf.gather(item_h, batch_item_indices)
                # embedded_neg_items = tf.gather(item_h, batch_neg_item_indices)

                embedded_neg_items = tf.gather(item_h, batch_neg_item_indices)

                query = tf.expand_dims(embedded_users, axis=-1)

                keys = tf.concat([
                    tf.expand_dims(embedded_items, axis=1),
                    embedded_neg_items
                ], axis=1)

                logits = tf.squeeze(keys @ query, axis=-1)
                # logits = tf.squeeze(logits, axis=-1) / 0.07
                mf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=tf.zeros(tf.shape(logits)[0], dtype=tf.int32)
                )

                # pos_logits = tf.reduce_sum(embedded_users * embedded_items, axis=-1)
                # neg_logits = tf.reduce_sum(embedded_users * embedded_neg_items, axis=-1)
                #
                # logits = tf.stack([pos_logits, neg_logits], axis=-1)
                # mf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                #     logits=logits,
                #     labels=tf.zeros_like(pos_logits, dtype=tf.int64)
                # )

                # mf_losses = tf.nn.softplus(-(pos_logits - neg_logits))

                # kernel_vars = [var for var in tape.watched_variables() if "kernel" in var.name]
                embedding_vars = [user_h, item_h]

                # kernel_l2_losses = [tf.nn.l2_loss(var) for var in kernel_vars]
                embedding_l2_losses = [tf.nn.l2_loss(var) for var in embedding_vars]

                # kernel_l2_loss = tf.add_n(kernel_l2_losses)
                embedding_l2_loss = tf.add_n(embedding_l2_losses)
                l2_loss = embedding_l2_loss * z_l2_coef  # + kernel_l2_loss * k_l2_coef
                loss = tf.reduce_sum(mf_losses) + l2_loss

            vars = tape.watched_variables()
            grads = tape.gradient(loss, vars)
            optimizer.apply_gradients(zip(grads, vars))

            return loss, mf_losses, l2_loss

        # summary_writer = tf.summary.create_file_writer("logs/infonce_dropout_os_item_appnp_lr_{}".format(lr))

        interval = 20

        time_spend = 0

        for epoch in range(0, num_epochs + 1):
            if epoch % interval == 0:
                user_h, item_h = forward(training=False)

                print("\nEvaluation before epoch {} ......".format(epoch))
                mean_results_dict = evaluate_mean_global_metrics(self.test_user_items_dict, self.train_user_items_dict,
                                                                 user_h, item_h, k_list=[5, 10, 15, 20],
                                                                 metrics=["precision", "recall", "ndcg"], )
                print(mean_results_dict)
                print()

                data = mean_results_dict.copy()
                data["epoch"] = epoch
                data["time"] = time_spend

                with open(result_path, "a", encoding="utf-8") as f:
                    f.write("{}\n".format(json.dumps(data)))

                # if epoch > 0:
                #     with summary_writer.as_default():
                #         tf.summary.scalar("ndcg", mean_results_dict["ndcg@20"], step=epoch // interval)

            step_losses = []
            step_mf_losses_list = []
            step_l2_losses = []

            start_time = time.time()

            for step, batch_edges in enumerate(
                    tf.data.Dataset.from_tensor_slices(self.train_user_item_edges).shuffle(
                        len(self.train_user_item_edges)).batch(batch_size)):
                batch_user_indices = batch_edges[:, 0]
                batch_item_indices = batch_edges[:, 1]
                # batch_neg_item_indices = np.random.randint(0, num_items, batch_item_indices.shape)
                # batch_neg_item_indices = tf.random.uniform(batch_item_indices.shape, 0, num_items, dtype=tf.int32)

                loss, mf_losses, l2_loss = train_step(batch_user_indices, batch_item_indices)

                step_losses.append(loss.numpy())
                step_mf_losses_list.append(mf_losses.numpy())
                step_l2_losses.append(l2_loss.numpy())

            end_time = time.time()

            time_spend += (end_time - start_time)

            if optimizer.learning_rate.numpy() > 1e-5:
                # optimizer.learning_rate.assign(optimizer.learning_rate * 0.995)
                optimizer.learning_rate.assign(optimizer.learning_rate * lr_decay)
                lr_status = "update lr => {:.4f}".format(optimizer.learning_rate.numpy())
            else:
                lr_status = "current lr => {:.4f}".format(optimizer.learning_rate.numpy())

            print("epoch = {}\tloss = {:.4f}\tmf_loss = {:.4f}\tl2_loss = {:.4f}\t{}\tepoch_time = {:.4f}s".format(
                epoch, np.mean(step_losses), np.mean(np.concatenate(step_mf_losses_list, axis=0)),
                np.mean(step_l2_losses), lr_status, end_time - start_time))

            if epoch == 1:
                print("the first epoch may take a long time to compile tf.function")


task = CFTask()
task.run()
