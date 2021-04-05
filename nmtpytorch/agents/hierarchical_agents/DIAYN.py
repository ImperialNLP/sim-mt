"""
File modified by Julia Ive julia.ive84@gmail.com
https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/hierarchical_agents/DIAYN.py

Original work Copyright (c) 2021 Petros Christodoulou
"""

import torch
from gym import Wrapper, spaces
from torch import optim, nn
import numpy as np
import random
import time
import copy
import torch.nn.functional as F
from ..Base_Agent import Base_Agent
from ..actor_critic_agents.SAC_Discrete import SAC_Discrete
from gym.utils import seeding
from ...utils.device import DEVICE
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


# NOTE: DIAYN calculates diversity of states penalty each timestep but it might be better to only base it on where the
# agent got to in the last timestep, or after X timesteps
# NOTE another problem with this is that the discriminator is trained from online data as it comes in which isn't iid
# so we could probably make it perform better by maintaining a replay buffer and using that to train the discriminator instead

class DIAYN(Base_Agent):
    """Hierarchical RL agent based on the paper Diversity is all you need (2018) - https://arxiv.org/pdf/1802.06070.pdf.
    Works in two stages:
        1) First it trains an agent that tries to reach different states depending on which skill number is
           inputted
        2) Then it trains an agent to maximise reward using its choice of skill for the lower level agent"""
    agent_name = "DIAYN"

    def __init__(self, config, actor=None, actor_optimizer=None):
        super().__init__(config)
        self.training_mode = True
        self.num_skills = config.hyperparameters["DIAYN"]["num_skills"]
        self.unsupervised_episodes = config.hyperparameters["DIAYN"]["num_unsupervised_episodes"]
        self.supervised_episodes = config.num_episodes_to_run - self.unsupervised_episodes

        assert self.hyperparameters["DIAYN"]["DISCRIMINATOR"][
                   "final_layer_activation"] == None, "Final layer activation for disciminator should be None"

        self.discriminator = self.create_NN(self.state_size, self.num_skills, key_to_use="DISCRIMINATOR")

        self.discriminator.to(DEVICE)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(),
                                                  lr=self.hyperparameters["DIAYN"]["DISCRIMINATOR"]["learning_rate"])
        self.agent_config = copy.deepcopy(config)
        self.agent_config.environment = DIAYN_Skill_Wrapper(copy.deepcopy(self.environment), self.num_skills, self)
        self.agent_config.hyperparameters = self.agent_config.hyperparameters["DIAYN"]["AGENT"]
        self.agent_config.hyperparameters["do_evaluation_iterations"] = False
        self.agent = SAC_Discrete(self.agent_config, actor=actor,
                                  actor_optimizer=optim.Adam(actor_optimizer,
                                                  lr=self.hyperparameters["Actor_Critic_Agents"]["Actor"]["learning_rate"]))  # We have to use SAC because it involves maximising the policy's entropy over actions which is also a part of DIAYN

        #self.timesteps_to_give_up_control_for = self.hyperparameters["DIAYN"]["MANAGER"][
        #    "timesteps_to_give_up_control_for"]
        self.batch = None

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True,
                       batch=None):
        start = time.time()
        self.agent.run_n_episodes(num_episodes=self.unsupervised_episodes, show_whether_achieved_goal=False,
                                  batch=batch)
        time_taken = time.time() - start
        pretraining_results = 0.0
        return pretraining_results, pretraining_results, time_taken

    def disciminator_learn(self, skill, discriminator_outputs):
        if not self.training_mode: return
        loss = nn.CrossEntropyLoss()(discriminator_outputs, torch.Tensor(skill).long().squeeze(-1).to(DEVICE))
        self.take_optimisation_step(self.discriminator_optimizer, self.discriminator, loss,
                                    self.hyperparameters["DIAYN"]["DISCRIMINATOR"]["gradient_clipping_norm"])

    def get_predicted_probability_of_skill(self, skill, action=None):
        """Gets the probability that the disciminator gives to the correct skill"""

        next_state = torch.Tensor(action).long()
        next_state_emb = self.agent.actor_local.dec.emb(next_state.to(DEVICE))
        # JI: discriminator will take the source and the latest action
        predicted_probabilities_unnormalised = self.discriminator(
            torch.cat((next_state_emb.detach(), self.agent.state_dict_in['src'][0][-1].detach()), dim=-1))
        predicted_probabilities_normalised = F.softmax(predicted_probabilities_unnormalised)
        probability_of_correct_skill = []
        for i, skill_example in enumerate(skill):
            skill_example = skill_example[0]
            probability_of_correct_skill.append(
                predicted_probabilities_normalised[i][skill_example].detach().cpu().numpy())

        return np.array(probability_of_correct_skill), predicted_probabilities_unnormalised


class DIAYN_Skill_Wrapper(object):
    """Open AI gym wrapper to help create a pretraining environment in which to train diverse skills according to the
    specification in the Diversity is all you need (2018) paper """

    def __init__(self, env, num_skills, meta_agent):
        self.num_skills = num_skills
        self.meta_agent = meta_agent
        self.prior_probability_of_skill = 1.0 / self.num_skills  # Each skill equally likely to be chosen
        self._max_episode_steps = env.max_steps
        self.environment_name = "DIAYN_Skill_Wrapper"
        self.action_space = env.action_space
        self.batch_length = env.batch_length
        self.env = env
        self.batch = None

    def reset(self, batch=None, **kwargs):
        self.batch_length = batch.size
        observation = self.env.reset(batch=batch)
        self.skill = np.array([random.randint(0, self.num_skills - 1) for i in range(self.env.batch_length)])
        self.skill = self.skill.reshape((-1, 1))
        return self.observation(observation)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def observation(self, observation):
        return np.concatenate((np.array(observation), np.array(self.skill)), axis=1)

    def step(self, action, batch=None, update_disc=True):

        next_state, _, done, _ = self.env.step(action)
        # JI: uncomment this to use unsupervised reward
        # new_reward = self.get_BLEU_reward(batch)
        new_reward, discriminator_outputs = self.calculate_new_reward(action=action)
        if update_disc:
            self.meta_agent.disciminator_learn(self.skill, discriminator_outputs)

        return self.observation(next_state), new_reward, done, _

    def get_BLEU_reward(self, batch):
        new_reward = []
        chencherry = SmoothingFunction()
        ref = batch['trg'].permute(1, 0)
        ref_lines = ref.cpu().tolist()
        dones = np.transpose(np.array(self.env.done))
        obs_list = self.env.obs
        obs = np.transpose(np.array(obs_list))
        hyp = obs * (1.0 - dones[:, :-1])
        # JI: compute the per sentence reward
        for i, ref_line in enumerate(ref_lines):
            ref_words = self.meta_agent.agent.actor_local.trg_vocab.idxs_to_sent(ref_line, debug=True)
            ref_ar = ref_words.split()
            new_real = ref_ar[1:]
            hyp_line = hyp[i][1:]
            # JI: pad hypotheses to not receive length penalty
            new_fake = np.full((len(new_real),), '<pad>', dtype='object')
            hyp_words = self.meta_agent.agent.actor_local.trg_vocab.idxs_to_sent(hyp_line, debug=True)
            hyp_ar = np.array(hyp_words.split())
            n = min(len(new_real), len(hyp_ar))
            # JI: fill in the so far generated words
            new_fake[:n] = hyp_ar[:n]
            local_score = sentence_bleu([new_real], new_fake, smoothing_function=chencherry.method5)
            new_reward.append(float(local_score))
        new_reward = np.array(new_reward)
        return new_reward

    def calculate_new_reward(self, action=None):
        """Calculates an intrinsic reward that encourages maximum exploration. It also keeps track of the discriminator
        outputs so they can be used for training"""
        self.skill = np.array([random.randint(0, self.num_skills - 1) for i in range(action.shape[0])])
        self.skill = self.skill.reshape((-1, 1))
        probability_correct_skill, disciminator_outputs = self.meta_agent.get_predicted_probability_of_skill(self.skill,
                                                                                                             action=action)

        # JI: get the difference between real and predicted skill
        new_reward = np.log(probability_correct_skill + 1e-8) - np.log(self.prior_probability_of_skill)
        return new_reward, disciminator_outputs
