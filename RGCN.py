import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from RGNN import RGNN


class RGCN(RGNN): #2 layers
    """@ Modeling Relational Data with Graph Convolutional Networks"""
    
    def __init__(self, args):
        super().__init__(args)
        print('\n\n' + '==' * 4 + ' < R-GCN > && < {} >'. \
              format(self.dataset) + '==' * 4)         
        self.load_data()
    

    def get_A(self):
        """Get adjacency matrix for each relations, normalized to row sum 1."""
        
        A = []
        for r in range(self.n_R):
            edges = self.KG[self.KG[:, 1] == r]
            edges = np.delete(edges, 1, 1)
            row, col = np.transpose(edges)
            count_dict = {x: list(row).count(x) for x in set(row)}
            data = np.array([1 / count_dict[x] for x in row])
            a = sp.coo_matrix((data, (row, col)), shape = (self.n_E, self.n_E))
            A.append((np.vstack((a.row, a.col)).transpose(), a.data, a.shape))
        return A
    
    
    def rgnn_layer(self):
        """A layer of R-GCN."""
        
        self.supports = [tf.sparse_placeholder(tf.float32)
                         for _ in range(self.n_R)]
        self.feed_dict = {self.supports[r]: self.A[r] for r in range(self.n_R)}
        
        with tf.variable_scope('R-GCN'):
            with tf.variable_scope('layer1'):
                h_out = self.rgcn_layer(None, self.n_E, self.h_dim, tf.nn.relu)
            with tf.variable_scope('layer2'):
                output = self.rgcn_layer(h_out, self.h_dim, self.out_dim)
                            
        return output
        
            
    def rgcn_layer(self, X, in_dim, out_dim, act = None):
        """
        A layer of R-GCN.
        If n_B == 0: don't apply basis decompasition.
        """
        
        K = np.sqrt(6.0 / (in_dim + out_dim))
        
        if self.n_B == 0:
            r_w = tf.get_variable('relation_weight', initializer = \
                  tf.random_uniform([self.n_R, in_dim, out_dim], -K, K))
        else:
            r_c = tf.get_variable('relation_coefficient', initializer = \
                  tf.random_uniform([self.n_R, self.n_B], -K, K))
            r_b = tf.get_variable('relation_basis', initializer = \
                  tf.random_uniform([self.n_B, in_dim, out_dim], -K, K))
            r_w = tf.reshape(tf.matmul(r_c, tf.reshape(r_b, [-1, in_dim * \
                  out_dim])), [-1, in_dim, out_dim])
        s_w = tf.get_variable('self_weight', initializer = \
              tf.random_uniform([in_dim, out_dim], -K, K))
        
        if X is None:
            out = s_w
        else:
            out = tf.matmul(X, s_w)
        
        for r in range(self.n_R):
            if X is None:
                adj_out = tf.sparse_tensor_dense_matmul(self.supports[r],
                                                        r_w[r])
            else:
                adj_out = tf.matmul(tf.sparse_tensor_dense_matmul( \
                          self.supports[r], X), r_w[r])
            out = out + adj_out

        if act:
            out = act(out)
            
        return tf.nn.dropout(out, self.keep)