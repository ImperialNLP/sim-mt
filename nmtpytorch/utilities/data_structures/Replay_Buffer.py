"""
Taken from https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
Original work Copyright (c) 2021 Petros Christodoulou
"""
from collections import namedtuple, deque
import random
import torch
import numpy as np
import sys
import traceback

class Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""
    
    def __init__(self, buffer_size, batch_size, seed):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "ref", "pos", "src"])
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, states, actions, rewards, next_states, dones, refs, poss, srcs):
        """Adds experience(s) into the replay buffer"""
        '''
        if True:
        #if type(dones) == list:
            
            assert type(dones[0]) != list, "A done shouldn't be a list"
            for i in range(64):
                print(actions[i])
                assert not isinstance(actions[i], int), "A done shouldn't be a list"
            print(actions)
            if len(actions) != 64:
                print('mumu')
                exit()
        
            experiences = [self.experience(state, action, reward, next_state, done, ref, pos, src)
                           for state, action, reward, next_state, done, ref, pos, src in
                           zip(states, actions, rewards, next_states, dones, refs, poss, srcs)]
            self.memory.extend(experiences)
        '''
        if True:
            experience = self.experience(states, actions, rewards, next_states, dones, refs, poss, srcs)
            print(sys._getframe().f_back.f_code.co_name)
            track = traceback.format_exc()
            #print(track)
            #exit()
            self.memory.append(experience)
        
    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        return experiences
        #print(len(experiences[0].state))
        #exit()
        '''
        if separate_out_data_types:
            states, actions, rewards, next_states, dones, refs, poss, srcs = self.separate_out_data_types(experiences)
            return states, actions, rewards, next_states, refs, poss, dones, srcs
        else:
            return experiences
        '''
            
    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        #print([e.action for e in experiences if e is not None])
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)
        #for e in experiences:
        #    print(e.ref)
        #exit()
        refs = torch.from_numpy(np.vstack([e.ref for e in experiences if e is not None])).float().to(self.device)
        poss = torch.from_numpy(np.vstack([int(e.pos) for e in experiences if e is not None])).float().to(self.device)
        #print(experiences[0].state)
        #print(experiences[0].src)
        #exit()
        srcs = torch.from_numpy(np.vstack([e.src for e in experiences if e is not None])).float().to(self.device)
        return states, actions, rewards, next_states, dones, refs, poss, srcs
    
    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None: batch_size = num_experiences
        else: batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)
