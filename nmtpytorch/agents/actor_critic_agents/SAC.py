"""
File modified by Julia Ive julia.ive84@gmail.com
https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC.py
Original work Copyright (c) 2021 Petros Christodoulou
"""
from ..Base_Agent import Base_Agent
from ...utilities.OU_Noise import OU_Noise
from ...utilities.data_structures.Replay_Buffer import Replay_Buffer
from torch.optim import Adam
import torch
from torch.distributions import Normal
import numpy as np
from ...utils.device import DEVICE

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6


class SAC(Base_Agent):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github implementation
      https://github.com/pranz24/pytorch-soft-actor-critic. It is an actor-critic algorithm where the agent is also trained
      to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SAC"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert self.action_types == "CONTINUOUS", "Action types must be continuous. Use SAC Discrete instead for discrete actions"
        assert self.config.hyperparameters["Actor"][
                   "final_layer_activation"] != "Softmax", "Final actor layer must not be softmax"
        self.hyperparameters = config.hyperparameters

        # JI: critic are overridden by SAC discrete
        self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                           key_to_use="Critic")
        self.critic_local_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                             key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                            key_to_use="Critic")
        self.critic_target_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                              key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed)
        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size * 2,
                                          key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]

        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.environment.action_space.shape).to(
                self.device)).item()  # heuristic value from the paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
                                  self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

    def save_result(self):
        """Saves the result of an episode of the game. Overriding the method in Base Agent that does this because we only
        want to keep track of the results during the evaluation episodes"""
        if self.episode_number == 1 or not self.do_evaluation_iterations:
            self.game_full_episode_scores.extend([self.total_episode_score_so_far])
            self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
            self.save_max_result_seen()

        elif (self.episode_number - 1) % TRAINING_EPISODES_PER_EVAL_EPISODE == 0:
            self.game_full_episode_scores.extend(
                [self.total_episode_score_so_far for _ in range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.rolling_results.extend(
                [np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]) for _ in
                 range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.save_max_result_seen()

    def reset_game(self, batch=None, epoch_end=False):
        """Resets the game information so we are ready to play a new episode"""
        if epoch_end:
            print('critic 1')
            print(self.critic1_gl_loss / self.gl_step_count)
            print('critic 2')
            print(self.critic2_gl_loss / self.gl_step_count)
            print('actor')
            print(self.actor_gl_loss / self.gl_step_count)

            self.critic1_gl_loss = 0.0
            self.critic2_gl_loss = 0.0
            self.actor_gl_loss = 0.0
            self.gl_step_count = 1

        Base_Agent.reset_game(self, batch=batch)
        if self.add_extra_noise: self.noise.reset()

        Base_Agent.reset_game(self, batch=batch)
        if self.add_extra_noise: self.noise.reset()

    def step(self, batch=None):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        # JI: first action is the <bos> symbol
        self.action = self.actor_local.get_bos(self.environment.batch_length).detach().cpu().numpy()
        prev_hid_state = None
        ref_batch = batch['trg'].permute(1, 0).cpu().detach().numpy()
        seq_reward = []
        self.replay_mode = False
        src_batch_exp = batch['src'].permute(1, 0)
        enc_args = {}
        # JI: get the source examples and detach
        # we take them as is and do not backpropagate
        self.state_dict_in = self.actor_local.encode(batch, **enc_args)
        self.state_dict_in_detached = self.state_dict_in
        self.state_dict_in_detached['src'][0].detach()
        self.state_dict_in_detached['src'][1].detach()

        # JI: we do not loop over the generated examples longer than 50 tokens
        while self.episode_step_number_val < min(ref_batch.shape[1] - 1 + 10, 50):
            self.episode_step_number_val += 1
            self.action, prev_hid_state = self.pick_action(eval_ep, batch=batch, prev_hid_state=prev_hid_state,
                                                              action=self.action)

            # JI we always update the discriminator
            update_disc = True
            self.conduct_action(self.action, batch=batch, update_disc=update_disc)

            seq_reward.append(self.reward)
            self.state = self.next_state
            self.global_step_number += 1

        # JI: we update Actor after certain scores
        # JI: uncomment for supervised reward

        # if len(self.game_full_episode_scores) > 4800:
        if True:
            update_actor = True
        else:
            update_actor = False

        if self.time_for_critic_and_actor_to_learn():
            for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                self.replay_mode = True
                self.learn(update_actor=update_actor)

        if not eval_ep:
            src_batch_exp = src_batch_exp.cpu().detach().numpy()

            # JI compute the TD reward
            dones = 1.0 * np.transpose(np.array(self.environment.env.done[:-1]))
            out_len = torch.Tensor(np.array(self.state[:, :-1]) * (1.0 - dones)).permute(1, 0).gt(2).float().sum(
                0).clamp(min=1)
            ref_len = torch.Tensor(np.array(ref_batch)).permute(1, 0).gt(2).float().sum(0).clamp(min=1)
            len_penalty = torch.abs(out_len - ref_len)

            seq_reward = np.transpose(np.array(seq_reward))
            R = seq_reward
            seq_reward[:, 1:] -= R[:, :-1]
            seq_reward = seq_reward - 0.0001 * np.tile(np.expand_dims(len_penalty, axis=-1), (1, seq_reward.shape[1]))
            seq_reward *= (1.0 - dones[:, 1:])
            # JI: scale the reward, used only for supervised, e.g. BLEU
            # seq_reward = seq_reward * np.tile(np.expand_dims((1 / self.alpha), axis=-1), (1, seq_reward.shape[1]))

            # JI: save sample in the replay buffer
            self.save_experience(experience=(
                self.state, self.action, seq_reward, self.next_state, dones[:, 1:], np.array(ref_batch),
                np.full((batch.size,), self.episode_step_number_val), src_batch_exp))

        if eval_ep: self.print_summary_of_latest_evaluation_episode()
        self.episode_number += 1

    def pick_action(self, eval_ep, state=None, batch=None, prev_hid_state=None, action=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True  3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        if state is None: state = self.state
        if eval_ep:
            action, prev_hid_state = self.actor_pick_action(state=state, eval=True, batch=batch)
        else:
            action, prev_hid_state = self.actor_pick_action(state=state, batch=batch,
                                                                                 prev_hid_state=prev_hid_state,
                                                                                 action=action)
        if self.add_extra_noise:
            action += self.noise.sample()
        return action, prev_hid_state

    def actor_pick_action(self, state=None, eval=False, batch=None, prev_hid_state=None, action=None):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""
        if state is None: state = self.state
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1: state = state.unsqueeze(0)
        if eval == False:
            action, _, _, prev_hid_state = self.produce_action_and_action_info(state, batch=batch,
                                                                               prev_hid_state=prev_hid_state,
                                                                               action=action)
        else:
            with torch.no_grad():
                _, _, action, prev_hid_state = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action, prev_hid_state

    def produce_action_and_action_info(self, state, batch=None, prev_hid_state=None, action=None):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""

        return _, _, _, _

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.global_step_number > self.hyperparameters["min_steps_before_learning"] and \
               self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters[
                   "update_every_n_steps"] == 0

    def learn(self, update_actor=False):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        qf1_loss_gl = 0.0
        qf2_loss_gl = 0.0
        policy_loss_gl = None
        xe_loss_gl = 0.0

        prev_state_q = None
        prev_prev_state_q = None
        prev_state_policy = None
        prev_state_xe = None

        prev_state_next_q1 = None
        prev_state_next_q2 = None

        prev_state_q1_policy = None
        prev_state_q2_policy = None

        # JI: get the sample
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, ref_batch, pos_batch, src_batch = \
            self.sample_experiences()[0]

        src_batch = torch.LongTensor(src_batch)
        src_batch = src_batch.permute(1, 0).to(DEVICE)
        new_batch = dict()
        new_batch['src'] = src_batch
        enc_args = {}
        ent = 0.0

        prev_states = []

        # JI: since we sample a new batch from all the previous batches
        # it will be different from the incoming training batch
        # we update the source batch here and freeze it
        self.state_dict_in = self.actor_local.encode(new_batch, **enc_args)
        self.state_dict_in_detached = self.state_dict_in
        self.state_dict_in_detached['src'][0].detach()
        self.state_dict_in_detached['src'][1].detach()

        _, _, _, prev_state_q = self.produce_action_and_action_info(state_batch, batch=new_batch,
                                                                       prev_hid_state=prev_prev_state_q,
                                                                       action=state_batch[:, 0])

        ref_batch_torch = torch.LongTensor(ref_batch[:, 1:]).permute(1, 0).to(DEVICE)
        ref_init = {}
        ref_mask = torch.ones(1, ref_batch_torch.size()[1]).to(DEVICE)
        ref_init['src'] = (ref_batch_torch, ref_mask)

        # JI: encode the references for the critics
        ref_emb = self.actor_local.dec.emb(ref_batch_torch)
        ref_batch_enc_prep = self.ref_enc(ref_batch_torch.detach(), embedded=ref_emb.detach())
        ref_batch_enc = {'src': ref_batch_enc_prep}

        ref_batch_enc_prep1 = self.ref_enc1(ref_batch_torch.detach(), embedded=ref_emb.detach())
        ref_batch_enc1 = {'src': ref_batch_enc_prep1}

        ref_batch_enc_prep_target = self.ref_enc_target(ref_batch_torch.detach(), embedded=ref_emb.detach())
        ref_batch_enc_target = {'src': ref_batch_enc_prep_target}

        ref_batch_enc_prep1_target = self.ref_enc1_target(ref_batch_torch.detach(), embedded=ref_emb.detach())
        ref_batch_enc1_target = {'src': ref_batch_enc_prep1_target}

        # JI: we iterate so that always to have 2 steps before
        for i in range(state_batch.shape[1] - 2):

            mask = np.transpose(mask_batch)[i]
            mask = torch.LongTensor(mask).to(DEVICE)
            prev_states.append(prev_prev_state_q)

            step_reward = reward_batch[:, i]
            step_reward = np.tile(np.expand_dims(step_reward, axis=-1), (1, 6405))

            # JI: here we fill in the true Q estimates
            if i < state_batch.shape[1] - 3:
                true_q_values = reward_batch[:, i + 1]
            else:
                true_q_values = reward_batch[:, i]
            true_q_values = np.tile(np.expand_dims(true_q_values, axis=-1), (1, 6405))

            if i + 1 < ref_batch.shape[1] - 1:

                step_ref = ref_batch[:, i + 2]
                for n, ind in enumerate(step_ref):
                    true_q_values[n, ind] = 1

            step_reward = torch.FloatTensor(step_reward).to(DEVICE)
            true_q_values = torch.FloatTensor(true_q_values).to(DEVICE)

            # JI: we consider the next action is taken hence state_batch[:, i + 1]
            # here we send the ref_init reference batch inside
            next_q_value, new_prev_state_q, prev_state_next_q1, prev_state_next_q2 = self.calculate_target_value(
                state_batch[:, i + 1], step_reward,
                [ref_batch_enc_target, ref_batch_enc1_target], true_q_values,
                mask, new_batch, prev_hid_state=prev_state_q, batch=ref_init,
                prev_q1=prev_state_next_q1, prev_q2=prev_state_next_q2)

            prev_state_q = new_prev_state_q
            prev_prev_state_q = prev_state_q

            """
            # JI: here we minimise the loss between our Q-value and the target value function
            # JI: we consider the current action state_batch[:, i]
            # JI: uncomment this to use supervised reward
            """

            # qf1_loss, qf2_loss, prev_state_q1, prev_state_q2 = self.calculate_critic_losses(
            #     [ref_batch_enc, ref_batch_enc1], next_q_value, state_batch[:, i + 1], mask_batch,
            #    prev_action=state_batch[:, i], prev_q1=prev_state_q1, prev_q2=prev_state_q2, batch=ref_init)

            # qf1_loss_gl += qf1_loss
            # qf2_loss_gl += qf2_loss

            # JI: we consider the current action state_batch[:, i]
            policy_loss, log_pi, prev_state_policy, prev_state_q1_policy, prev_state_q2_policy = self.calculate_actor_loss(
                [ref_batch_enc, ref_batch_enc1], src_batch=new_batch, batch=ref_init,
                true_q_values=next_q_value, prev_hid_state=prev_state_policy, prev_action=state_batch[:, i],
                prev_q1=prev_state_q1_policy, prev_q2=prev_state_q2_policy, mask=mask)

            if i < ref_batch.shape[1] - 1:
                xe_action, (xe_action_probabilities,
                            xe_log_action_probabilities), xe_max_act, prev_state_xe = self.produce_action_and_action_info(
                    src_batch=src_batch, batch=new_batch, ref_batch=None,
                    action=ref_batch[:, i], prev_hid_state=prev_state_xe)

                xe_loss_gl += self.nll_loss(xe_log_action_probabilities,
                                            ref_batch_torch[i, :].long().to(DEVICE)) * ref_batch_torch[i, :].long().to(
                    DEVICE).ne(0).float()

            if policy_loss_gl == None:
                policy_loss_gl = policy_loss.sum(1, keepdim=True)
            else:
                policy_loss_gl = torch.cat((policy_loss_gl, policy_loss.sum(1, keepdim=True)), 1)

            if self.automatic_entropy_tuning:
                alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
            else:
                alpha_loss = None

        # qf1_loss_gl = qf1_loss_gl.mean()
        # qf2_loss_gl = qf2_loss_gl.mean()

        policy_loss_gl = policy_loss_gl.sum(0).mean()
        xe_loss_gl = xe_loss_gl.mean()

        self.update_all_parameters(qf1_loss_gl, qf2_loss_gl, policy_loss_gl, alpha_loss, xe_loss=xe_loss_gl,
                                   update_actor=update_actor)

    def sample_experiences(self):
        return self.memory.sample()

    def calculate_critic_losses(self, ref_batch, next_q_value, action_batch, mask_batch, prev_action=None,
                                prev_q1=None, prev_q2=None, batch=None):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy term is taken into account"""
        return _, _, _, _

    def calculate_actor_loss(self, ref_batch, src_batch=None, batch=None,
                             true_q_values=None, prev_hid_state=None, prev_action=None,
                             prev_q1=None, prev_q2=None, mask=None):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        return _, _, _, _, _

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def update_all_parameters(self, critic_loss_1, critic_loss_2, actor_loss, alpha_loss, xe_loss=None,
                              update_actor=False):
        """Updates the parameters for the actor, both critics and (if specified) the temperature parameter"""

        # self.take_optimisation_step(self.critic_optimizer, [self.critic_local, self.ref_enc], critic_loss_1,
        #                             self.hyperparameters["Critic"]["gradient_clipping_norm"])
        # self.take_optimisation_step(self.critic_optimizer_2, [self.critic_local_2, self.ref_enc1], critic_loss_2,
        #                            self.hyperparameters["Critic"]["gradient_clipping_norm"])

        if update_actor:
            self.actor_optimizer.zero_grad()

            if xe_loss != None:
                actor_loss = 0.01 * actor_loss + 0.1 * xe_loss

            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

        # self.soft_update_of_target_network(self.ref_enc, self.ref_enc_target,
        #                                   self.hyperparameters["Critic"]["tau"])
        # self.soft_update_of_target_network(self.ref_enc1, self.ref_enc1_target,
        #                                    self.hyperparameters["Critic"]["tau"])

        # self.soft_update_of_target_network(self.critic_local, self.critic_target,
        #                                    self.hyperparameters["Critic"]["tau"])
        # self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
        #                                   self.hyperparameters["Critic"]["tau"])
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

        # self.critic1_gl_loss += critic_loss_1.detach()
        # self.critic2_gl_loss += critic_loss_2.detach()
        self.actor_gl_loss += actor_loss.detach()
        self.gl_step_count += 1

    def print_summary_of_latest_evaluation_episode(self):
        """Prints a summary of the latest episode"""
        print(" ")
        print("----------------------------")
        print("Episode score {} ".format(self.total_episode_score_so_far))
        print("----------------------------")
