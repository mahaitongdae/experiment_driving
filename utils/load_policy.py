#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/30
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: load_policy.py
# =====================================
import argparse
import json

import tensorflow as tf
import numpy as np

from utils.policy import Policy4Toyota
from utils.preprocessor import Preprocessor


class LoadPolicy(object):
    def __init__(self, exp_dir, iter):
        model_dir = exp_dir
        parser = argparse.ArgumentParser()
        params = json.loads(open(exp_dir + '/config.json').read())
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        self.args = parser.parse_args()
        self.policy = Policy4Toyota(self.args)
        self.policy.load_weights(model_dir, iter)
        self.preprocessor = Preprocessor((self.args.obs_dim,), self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         gamma=self.args.gamma)
        # self.preprocessor.load_params(load_dir)

    @tf.function
    def run(self, obs):
        processed_obs = self.preprocessor.np_process_obses(obs)
        action, logp = self.policy.compute_action(processed_obs[np.newaxis, :])
        return action[0]

    @tf.function
    def values(self, obs):
        processed_obs = self.preprocessor.np_process_obses(obs)
        values = self.policy.compute_vs(processed_obs[np.newaxis, :])
        return values[0]
