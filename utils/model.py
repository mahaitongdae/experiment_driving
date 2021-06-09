#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: model.py
# =====================================

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention

from utils.model_utils import veh_positional_encoding, EncoderLayer

tf.config.experimental.set_visible_devices([], 'GPU')


class MLPNet(Model):
    def __init__(self, input_dim, num_hidden_layers, num_hidden_units, hidden_activation, output_dim, **kwargs):
        super(MLPNet, self).__init__(name=kwargs['name'])
        self.first_ = Dense(num_hidden_units,
                            activation=hidden_activation,
                            kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                            dtype=tf.float32)
        self.hidden = Sequential([Dense(num_hidden_units,
                                        activation=hidden_activation,
                                        kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                                        dtype=tf.float32) for _ in range(num_hidden_layers-1)])
        output_activation = kwargs['output_activation'] if kwargs.get('output_activation') else 'linear'
        self.outputs = Dense(output_dim,
                             activation=output_activation,
                             kernel_initializer=tf.keras.initializers.Orthogonal(1.),
                             bias_initializer=tf.keras.initializers.Constant(0.),
                             dtype=tf.float32)
        self.build(input_shape=(None, input_dim))

    def call(self, x, **kwargs):
        x = self.first_(x)
        x = self.hidden(x)
        x = self.outputs(x)
        return x

class AttnNet(Model):
    def __init__(self, ego_dim, total_veh_dim, veh_num, tracking_dim,
                 num_attn_layers, d_model, d_ff, num_heads, dropout,
                 max_len=10, **kwargs):
        super(AttnNet, self).__init__(name=kwargs['name'])

        assert total_veh_dim % veh_num == 0
        self.ego_dim = ego_dim
        self.veh_num = veh_num
        self.veh_dim = total_veh_dim // veh_num
        self.tracking_dim = tracking_dim

        self.num_layers = num_attn_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.dropout_rate = dropout

        self.ego_embedding = Sequential([tf.keras.layers.InputLayer(input_shape=(self.ego_dim+self.tracking_dim,)),
                                         Dense(units=d_model,
                                               kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                                               dtype=tf.float32)])
        self.vehs_embedding = Sequential([tf.keras.layers.InputLayer(input_shape=(self.veh_dim,)),
                                          Dense(units=d_model,
                                                kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
                                                dtype=tf.float32)])

        self.pe = veh_positional_encoding(max_len, d_model)
        self.dropout = Dropout(self.dropout_rate)

        self.attn_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout)
                            for _ in range(self.num_layers-1)]
        self.out_attn = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.build(input_shape=[(None, 1, ego_dim+tracking_dim), (None, veh_num, self.veh_dim),
                                (None, veh_num+1, veh_num+1), (None, veh_num+1, veh_num+1)]) # todo: variant length


    def call(self, input, **kwargs):
        training = kwargs.get('training')
        x_ego, x_vehs, padding_mask, mu_mask = input[0], input[1], input[2], input[3]
        assert x_ego.shape[2] == self.ego_dim+self.tracking_dim
        assert x_vehs.shape[2] == self.veh_dim
        assert x_vehs.shape[1] == self.veh_num

        seq_len = x_ego.shape[1] + x_vehs.shape[1]
        x1 = self.ego_embedding(x_ego)
        x2 = self.vehs_embedding(x_vehs)
        x = tf.concat([x1, x2], axis=1)
        assert x.shape[1] == seq_len
        x += self.pe[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers-1):
            x = self.attn_layers[i](x, training, padding_mask)

        output_mask = tf.maximum(padding_mask, mu_mask)
        x, attn_weights = self.out_attn(x, x, attention_mask=output_mask,
                                        return_attention_scores=True, training=training)

        return x, attn_weights

