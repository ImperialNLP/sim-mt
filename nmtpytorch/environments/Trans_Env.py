"""
Class for the Translation environment
"""

import numpy as np
from ..utils.device import DEVICE
import torch

class Trans_Env(object):

    def __init__(self, environment_name):
        self.batch_size = 64
        self.batch_length = self.batch_size
        self.seq_len = 10
        #IWSLT
        #self.vocab_size = 22811
        #FR
        #self.vocab_size = 5874
        #DE
        self.vocab_size = 6405
        self.environment_name = environment_name
        self.env = environment_name
        self.action_space = torch.zeros(self.vocab_size, device=DEVICE)
        self.steps_taken = 1
        self.max_steps = 55
        self._max_episode_steps = self.max_steps
        self.obs = []
        self.obs.append(np.ones([self.batch_size,]))
        self.done = []
        self.done.append(np.full((self.batch_length,), False))
        self.done.append(np.full((self.batch_length,), False))
        self.reward = np.zeros((self.batch_length,))

    def reset(self, batch=None):
        
        self.batch_size = batch.size
        self.batch_length = self.batch_size
        self.steps_taken = 1
        # We always assume a <bos> (1) state
        # see vocabulary.py for reserved tokens
        self.obs = []
        self.obs.append(np.ones([self.batch_size,]))
        self.done = []
        self.done.append(np.full((self.batch_length,), False))
        self.done.append(np.full((self.batch_length,), False))
        return np.transpose(np.array(self.obs))

    def step(self, action):
        self.obs.append(action)
        self.steps_taken += 1
        done_local = np.full((self.batch_length,), False)
        for n, action_step in enumerate(action):
            # if <pad> (0) or <eos> (2) we mark as Done (sentence is complete)
            if action_step == 2 or action_step == 0 or self.done[-1][n]:
                done_local[n] = True
        self.done.append(done_local)
        info = dict()
        return np.transpose(np.array(self.obs)), self.reward, np.transpose(np.array(self.done)), info
