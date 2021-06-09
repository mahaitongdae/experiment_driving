#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: policy.py
# =====================================

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PolynomialDecay
import numpy as np

from utils.model import MLPNet
from utils.model import AttnNet

NAME2MODELCLS = dict([('MLP', MLPNet),('Attn', AttnNet)])


class Policy4Toyota(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, args):
        super().__init__()
        self.args = args
        obs_dim, act_dim = self.args.obs_dim, self.args.act_dim
        n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
        value_model_cls, policy_model_cls = NAME2MODELCLS[self.args.value_model_cls], \
                                            NAME2MODELCLS[self.args.policy_model_cls]
        self.policy = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                       output_activation=self.args.policy_out_activation)
        policy_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='adam_opt')

        self.obj_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='obj_v')
        self.con_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='con_v')

        obj_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        self.obj_value_optimizer = self.tf.keras.optimizers.Adam(obj_value_lr_schedule, name='objv_adam_opt')

        con_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        self.con_value_optimizer = self.tf.keras.optimizers.Adam(con_value_lr_schedule, name='conv_adam_opt')

        self.models = (self.obj_v, self.con_v, self.policy,)
        self.optimizers = (self.obj_value_optimizer, self.con_value_optimizer, self.policy_optimizer)

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean

    def _logits2dist(self, logits):
        mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))
        if self.args.action_range is not None:
            act_dist = (
                self.tfp.distributions.TransformedDistribution(
                    distribution=act_dist,
                    bijector=self.tfb.Chain(
                        [self.tfb.Affine(scale_identity_multiplier=self.args.action_range),
                         self.tfb.Tanh()])
                ))
        return act_dist

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            if self.args.deterministic_policy:
                mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    @tf.function
    def compute_obj_v(self, obs):
        with self.tf.name_scope('compute_obj_v') as scope:
            return tf.squeeze(self.obj_v(obs), axis=1)

    @tf.function
    def compute_con_v(self, obs):
        with self.tf.name_scope('compute_con_v') as scope:
            return tf.squeeze(self.con_v(obs), axis=1)

class Policy4Lagrange(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, args):
        super().__init__()
        self.args = args
        obs_dim, act_dim = self.args.obs_dim, self.args.act_dim
        mu_dim = self.args.con_dim
        n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
        value_model_cls, policy_model_cls = NAME2MODELCLS[self.args.value_model_cls], \
                                            NAME2MODELCLS[self.args.policy_model_cls]
        mu_model_cls = NAME2MODELCLS[self.args.mu_model_cls]
        self.policy = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                       output_activation=self.args.policy_out_activation)
        policy_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='adam_opt')

        self.obj_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='obj_v')
        self.con_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='con_v')

        self.mu = mu_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, mu_dim, name='mu', output_activation=
                               self.args.mu_out_activation)

        mu_value_lr_schedule = PolynomialDecay(*self.args.mu_lr_schedule)
        self.mu_optimizer = self.tf.optimizers.Adam(mu_value_lr_schedule, name='mu_adam_opt')

        # obj_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        # self.obj_value_optimizer = self.tf.keras.optimizers.Adam(obj_value_lr_schedule, name='objv_adam_opt')
        #
        # con_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        # self.con_value_optimizer = self.tf.keras.optimizers.Adam(con_value_lr_schedule, name='conv_adam_opt')

        self.models = (self.policy, self.mu)
        self.optimizers = (self.policy_optimizer, self.mu) # self.obj_value_optimizer, self.con_value_optimizer,

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].set_weights(weight)

    @tf.function
    def apply_gradients(self, iteration, grads):
        policy_len = len(self.policy.trainable_weights)
        policy_grad, mu_grad = grads[:policy_len], grads[policy_len:]
        self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
        if iteration % self.args.mu_update_interval == 0:
            self.mu_optimizer.apply_gradients(zip(mu_grad, self.mu.trainable_weights))

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean

    def _logits2dist(self, logits):
        mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))
        if self.args.action_range is not None:
            act_dist = (
                self.tfp.distributions.TransformedDistribution(
                    distribution=act_dist,
                    bijector=self.tfb.Chain(
                        [self.tfb.Affine(scale_identity_multiplier=self.args.action_range),
                         self.tfb.Tanh()])
                ))
        return act_dist

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            if self.args.deterministic_policy:
                mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    # @tf.function
    # def compute_obj_v(self, obs):
    #     with self.tf.name_scope('compute_obj_v') as scope:
    #         return tf.squeeze(self.obj_v(obs), axis=1)
    #
    # @tf.function
    # def compute_con_v(self, obs):
    #     with self.tf.name_scope('compute_con_v') as scope:
    #         return tf.squeeze(self.con_v(obs), axis=1)

    @tf.function
    def compute_mu(self, obs):
        with self.tf.name_scope('compute_mu') as scope:
            return self.mu(obs)

class AttnPolicy4Lagrange(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, args):
        super().__init__()
        self.args = args

        obs_dim, act_dim = self.args.obs_dim, self.args.act_dim
        mu_dim = self.args.con_dim
        veh_dim = self.args.veh_dim
        veh_num = self.args.veh_num
        ego_dim = self.args.ego_dim
        tracking_dim = self.args.tracking_dim

        d_model = self.args.d_model
        num_attn_layers = self.args.num_attn_layers
        d_ff = self.args.d_ff
        num_heads = self.args.num_heads
        dropout = self.args.drop_rate
        max_len = self.args.max_veh_num

        assert tracking_dim + ego_dim + veh_dim*veh_num == obs_dim
        assert 4 + veh_num * 4 == mu_dim

        n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
        value_model_cls, policy_model_cls = NAME2MODELCLS[self.args.value_model_cls], \
                                            NAME2MODELCLS[self.args.policy_model_cls]
        backbone_cls = NAME2MODELCLS[self.args.backbone_cls]

        # Attention backbone
        self.backbone = backbone_cls(ego_dim, obs_dim-tracking_dim-ego_dim, veh_num, tracking_dim,
                                     num_attn_layers, d_model, d_ff, num_heads, dropout,
                                     max_len, name='backbone')
        mu_value_lr_schedule = PolynomialDecay(*self.args.mu_lr_schedule)
        self.mu_optimizer = self.tf.optimizers.Adam(mu_value_lr_schedule, name='mu_adam_opt')

        # self.policy = Sequential([tf.keras.layers.InputLayer(input_shape=(d_model,)),
        #                           Dense(d_model, activation=self.args.policy_out_activation,
        #                                 kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2.)),
        #                                 dtype=tf.float32),
        #                           Dense(act_dim * 2, activation=self.args.policy_out_activation,
        #                                 kernel_initializer=tf.keras.initializers.Orthogonal(1.),
        #                                 bias_initializer = tf.keras.initializers.Constant(0.),
        #                                 dtype = tf.float32),])
        self.policy = policy_model_cls(d_model, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                       output_activation=self.args.policy_out_activation)
        policy_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='adam_opt')

        # self.value = Sequential([tf.keras.Input(shape=(d_model,)),
        #                          Dense(1, activation='linear',
        #                                kernel_initializer=tf.keras.initializers.Orthogonal(1.),
        #                                bias_initializer=tf.keras.initializers.Constant(0.),
        #                                dtype=tf.float32),])
        # value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        # self.value_optimizer = self.tf.keras.optimizers.Adam(value_lr_schedule, name='v_adam_opt')

        self.models = (self.backbone, self.policy)
        self.optimizers = (self.mu_optimizer, self.policy_optimizer)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].set_weights(weight)

    @tf.function
    def apply_gradients(self, iteration, grads):
        policy_len = len(self.policy.trainable_weights)
        policy_grad, mu_grad = grads[:policy_len], grads[policy_len:]
        self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
        if iteration % self.args.mu_update_interval == 0:
            self.mu_optimizer.apply_gradients(zip(mu_grad, self.backbone.trainable_weights))

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean

    def _logits2dist(self, logits):
        mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))
        if self.args.action_range is not None:
            act_dist = (
                self.tfp.distributions.TransformedDistribution(
                    distribution=act_dist,
                    bijector=self.tfb.Chain(
                        [self.tfb.Affine(scale_identity_multiplier=self.args.action_range),
                         self.tfb.Tanh()])
                ))
        return act_dist

    @tf.function
    def compute_mu(self, obs, nonpadding_ind, training=True):
        def create_padding_mask(batch_size, seq_len, nonpadding_ind):
            nonpadding_ind = tf.cast(nonpadding_ind, dtype=tf.float32)
            nonpadding_ind = tf.concat([tf.ones((batch_size,1)), nonpadding_ind], axis=-1)
            nonpadding_ind = tf.reshape(nonpadding_ind, (batch_size, 1, -1))
            repaet_times = tf.constant([1, seq_len, 1], tf.int32)

            return tf.tile(nonpadding_ind, repaet_times)

        def create_mu_mask(batch_size, seq_len):
            mask = np.identity(seq_len, dtype=np.float32)
            mask[:, 0] = 1
            mask[0, :] = 1
            mask = mask[np.newaxis, :, :]
            return tf.convert_to_tensor(np.repeat(mask, repeats=batch_size, axis=0), dtype=tf.float32)

        with self.tf.name_scope('compute_mu') as scope:
            batch_size = (obs).shape[0]
            seq_len = self.args.veh_num+1
            x_ego = tf.expand_dims(obs[:, :self.args.ego_dim+self.args.tracking_dim], axis=1)
            x_vehs = tf.reshape(obs[:, self.args.ego_dim+self.args.tracking_dim:], (batch_size, -1, self.args.veh_dim))

            assert x_vehs.shape[1] == self.args.veh_num

            # hidden, attn_weights = self.backbone(x_ego, x_vehs,
            #                                      padding_mask=create_padding_mask(batch_size, seq_len, nonpadding_ind),
            #                                      mu_mask=create_mu_mask(batch_size, seq_len),
            #                                      training=training)
            hidden, attn_weights = self.backbone([x_ego, x_vehs,
                                                   create_padding_mask(batch_size, seq_len, nonpadding_ind),
                                                   create_mu_mask(batch_size, seq_len),],
                                                 training=training)
            mu_attn = attn_weights[:, :, 0, 1:]
            return hidden[:, 0, :], tf.cast(tf.exp(5*mu_attn)-1, dtype=tf.float32)

    @tf.function
    def compute_action(self, obs, nonpadding_ind, training=True):
        hidden, _ = self.compute_mu(obs, nonpadding_ind, training)
        hidden = tf.stop_gradient(hidden)
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(hidden)
            if self.args.deterministic_policy:
                mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    # @tf.function
    # def compute_v(self, hidden):
    #     with self.tf.name_scope('compute_v') as scope:
    #         return tf.squeeze(self.value(hidden), axis=1)
