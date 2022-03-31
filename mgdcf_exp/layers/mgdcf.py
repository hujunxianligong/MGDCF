# coding=utf-8
import numpy as np
import scipy.sparse as sp
import tf_sparse as tfs
from grecx.layers import LightGCN
from tf_geometric import SparseAdj


def build_homo_adjs(user_item_edges, num_users, num_items, adj_drop_rate):
    user_item_edge_index = user_item_edges.T
    user_item_row, user_item_col = user_item_edge_index
    user_item_adj = sp.csr_matrix((np.ones_like(user_item_row), (user_item_row, user_item_col)),
                                  shape=[num_users, num_items])
    item_user_adj = user_item_adj.T

    def convert_adj_to_trans(adj):
        deg = np.array(adj.sum(axis=-1)).flatten().astype(np.float32)
        inv_deg = np.power(deg, -1)
        inv_deg[np.isnan(inv_deg)] = 0.0
        inv_deg[np.isinf(inv_deg)] = 0.0

        trans = sp.diags(inv_deg) @ adj
        return trans

    # def convert_adj_to_trans(adj):
    #     def compute_deg(axis):
    #         deg = np.array(adj.sum(axis=axis)).flatten().astype(np.float32)
    #         inv_sqrt_deg = np.power(deg, -0.5)
    #         inv_sqrt_deg[np.isnan(inv_sqrt_deg)] = 0.0
    #         inv_sqrt_deg[np.isinf(inv_sqrt_deg)] = 0.0
    #         inv_sqrt_deg = sp.diags(inv_sqrt_deg)
    #         return inv_sqrt_deg
    #
    #     trans = compute_deg(axis=-1) @ adj @ compute_deg(axis=0)
    #     return trans

    user_item_trans = convert_adj_to_trans(user_item_adj)
    item_user_trans = convert_adj_to_trans(item_user_adj)

    def convert_hetero_to_homo(trans_ab, trans_ba):
        homo_trans = trans_ab @ trans_ba
        # homo_trans += homo_trans.T
        homo_trans = homo_trans.multiply(homo_trans.T)
        homo_trans.setdiag(0.0)

        # # renorm
        # homo_trans = convert_adj_to_trans(homo_trans)

        homo_trans = homo_trans.tocoo()

        probs = homo_trans.data
        probs = probs[probs > 0.0]
        sorted_probs = np.sort(probs)
        threshold = sorted_probs[int(len(probs) * adj_drop_rate)]

        mask = homo_trans.data > threshold
        homo_adj = sp.csr_matrix((np.ones_like(homo_trans.data[mask]), (homo_trans.row[mask], homo_trans.col[mask])),
                                 shape=homo_trans.shape)

        homo_adj = homo_adj.maximum(homo_adj.T)
        # homo_adj = homo_adj.minimum(homo_adj.T)
        homo_adj.eliminate_zeros()

        print(homo_trans.sum(), homo_adj.sum())
        print(len(homo_trans.nonzero()[0]), len(homo_adj.nonzero()[0]))
        #
        #
        # deg = homo_adj.sum(axis=-1)
        # degs = np.array(deg, dtype=np.int32).flatten()
        # counter = Counter(degs)
        # # for deg in range(np.max(deg) + 1):
        # #     print(deg, ": ", counter[deg])
        #
        # x = np.arange(0, np.max(deg) + 1)
        # y = [counter[deg] for deg in x]
        #
        # for deg, count in zip(x, y):
        #     print(deg, count)
        # # asdfasdf
        #
        # from matplotlib import pyplot as plt
        # plt.scatter(x, y)
        # # plt.fill_between(x, 0, y)
        # plt.show()
        # asdfasdf

        # homo_adj = sp.csr_matrix((homo_trans.data[mask], (homo_trans.row[mask], homo_trans.col[mask])),
        #                          shape=homo_trans.shape)

        return homo_adj

    user_user_adj = convert_hetero_to_homo(user_item_trans, item_user_trans).tocoo()
    item_item_adj = convert_hetero_to_homo(item_user_trans, user_item_trans).tocoo()
    return user_user_adj, item_item_adj






from tf_geometric.nn.conv.gcn import gcn_build_cache_for_graph, gcn_norm_adj
from tf_geometric.nn.conv.appnp import appnp
import tensorflow as tf
import warnings


def compute_gamma(alpha, beta, k):
    return np.power(beta, k) + alpha * np.sum([np.power(beta, i) for i in range(k)])



class HomoMGDN(tf.keras.Model):


    def __init__(self,
                 k=10,
                 alpha=0.1,
                 beta=0.9,
                 x_drop_rate=0.0,
                 edge_drop_rate=0.0,
                 z_drop_rate=0.0,
                 activation=None,
                 kernel_regularizer=None, bias_regularizer=None, *args, **kwargs):
        """

        :param units_list: List of Positive integers consisting of dimensionality of the output space of each dense layer.
        :param dense_activation: Activation function to use for the dense layers,
            except for the last dense layer, which will not be activated.
        :param activation: Activation function to use for the output.
        :param k: Number of propagation power iterations.
        :param alpha: Teleport Probability.
        :param dense_drop_rate: Dropout rate for the output of every dense layer (except the last one).
        :param last_dense_drop_rate: Dropout rate for the output of the last dense layer.
            last_dense_drop_rate is usually set to 0.0 for classification tasks.
        :param edge_drop_rate: Dropout rate for the edges/adj used for propagation.
        :param kernel_regularizer: Regularizer function applied to the `kernel` weights matrices.
        :param bias_regularizer: Regularizer function applied to the bias vectors.
        """

        super().__init__(*args, **kwargs)
        # self.units_list = units_list
        # self.dense_activation = dense_activation
        self.activation = activation
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = compute_gamma(alpha, beta, k)

        self.x_drop_rate = x_drop_rate
        self.edge_drop_rate = edge_drop_rate
        self.z_drop_rate = z_drop_rate

        self.x_dropout = tf.keras.layers.Dropout(x_drop_rate)
        self.z_dropout = tf.keras.layers.Dropout(z_drop_rate)

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.kernels = []
        self.biases = []

    def build_cache_for_graph(self, graph, override=False):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.

        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None
        """
        gcn_build_cache_for_graph(graph, override=override)

    def cache_normed_edge(self, graph, override=False):
        """
        Manually compute the normed edge based on this layer's GCN normalization configuration (self.renorm and self.improved) and put it in graph.cache.
        If the normed edge already exists in graph.cache and the override parameter is False, this method will do nothing.

        :param graph: tfg.Graph, the input graph.
        :param override: Whether to override existing cached normed edge.
        :return: None

        .. deprecated:: 0.0.56
            Use ``build_cache_for_graph`` instead.
        """
        warnings.warn(
            "'MGDCF.cache_normed_edge(graph, override)' is deprecated, use 'MGDCF.build_cache_for_graph(graph, override)' instead",
            DeprecationWarning)
        return self.build_cache_for_graph(graph, override=override)

    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """

        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        num_nodes = tfs.shape(x)[0]
        # updated_edge_index, normed_edge_weight = gcn_norm_edge(edge_index, num_nodes, edge_weight, cache=cache)
        sparse_adj = SparseAdj(edge_index, edge_weight, [num_nodes, num_nodes])
        normed_sparse_adj = gcn_norm_adj(sparse_adj, cache=cache) \
            .dropout(self.edge_drop_rate, training=training)


        h = self.x_dropout(x, training=training)

        output = h

        for i in range(self.k):
            output = normed_sparse_adj @ output
            output = output * self.beta + h * self.alpha

        if self.activation is not None:
            output = self.activation(output)

        output /= self.gamma

        output = self.z_dropout(output, training=training)

        return output



class HeteroMGDN(tf.keras.Model):


    def __init__(self,
                 k=10,
                 alpha=0.1,
                 beta=0.9,
                 x_drop_rate=0.0,
                 edge_drop_rate=0.0,
                 z_drop_rate=0.0,
                 activation=None,
                 kernel_regularizer=None, bias_regularizer=None, *args, **kwargs):
        """

        :param units_list: List of Positive integers consisting of dimensionality of the output space of each dense layer.
        :param dense_activation: Activation function to use for the dense layers,
            except for the last dense layer, which will not be activated.
        :param activation: Activation function to use for the output.
        :param k: Number of propagation power iterations.
        :param alpha: Teleport Probability.
        :param dense_drop_rate: Dropout rate for the output of every dense layer (except the last one).
        :param last_dense_drop_rate: Dropout rate for the output of the last dense layer.
            last_dense_drop_rate is usually set to 0.0 for classification tasks.
        :param edge_drop_rate: Dropout rate for the edges/adj used for propagation.
        :param kernel_regularizer: Regularizer function applied to the `kernel` weights matrices.
        :param bias_regularizer: Regularizer function applied to the bias vectors.
        """

        super().__init__(*args, **kwargs)
        # self.units_list = units_list
        # self.dense_activation = dense_activation
        self.activation = activation
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = compute_gamma(alpha, beta, k)

        self.x_drop_rate = x_drop_rate
        self.edge_drop_rate = edge_drop_rate
        self.z_drop_rate = z_drop_rate

        self.x_dropout = tf.keras.layers.Dropout(x_drop_rate)
        self.z_dropout = tf.keras.layers.Dropout(z_drop_rate)

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    @classmethod
    def build_virtual_edge_index(cls, user_item_edge_index, num_users=None):
        return LightGCN.build_virtual_edge_index(user_item_edge_index, num_users)

    @classmethod
    def norm_adj(cls, edge_index, num_nodes, cache=None):
        return LightGCN.norm_adj(edge_index, num_nodes, cache=cache)

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

    def call(self, inputs, cache=None, training=None, mask=None):
        """

        :param inputs: List of graph info: [x, edge_index, edge_weight]
        :param cache: A dict for caching A' for GCN. Different graph should not share the same cache dict.
        :return: Updated node features (x), shape: [num_nodes, units]
        """

        if len(inputs) == 3:
            x, edge_index, edge_weight = inputs
        else:
            x, edge_index = inputs
            edge_weight = None

        num_nodes = tfs.shape(x)[0]
        # updated_edge_index, normed_edge_weight = gcn_norm_edge(edge_index, num_nodes, edge_weight, cache=cache)
        # sparse_adj = SparseAdj(edge_index, edge_weight, [num_nodes, num_nodes])

        normed_sparse_adj = self.norm_adj(edge_index, num_nodes=num_nodes, cache=cache) \
            .dropout(self.edge_drop_rate, training=training)

        h = self.x_dropout(x, training=training)

        output = h

        for i in range(self.k):
            output = normed_sparse_adj @ output
            output = output * self.beta + h * self.alpha

        if self.activation is not None:
            output = self.activation(output)

        output /= self.gamma
        output = self.z_dropout(output, training=training)

        return output











