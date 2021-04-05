"""
File modified by Julia Ive julia.ive84@gmail.com
https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py
Original work Copyright (c) 2021 Petros Christodoulou
"""

import torch
from ...layers import TextEncoder
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from ..Base_Agent import Base_Agent
from ...utilities.data_structures.Replay_Buffer import Replay_Buffer
from .SAC import SAC
from ...utilities.Utility_Functions import create_actor_distribution
from ...utils.device import DEVICE
from torch import nn
from ...layers.decoders import get_decoder


class SAC_Discrete(SAC):
    """The Soft Actor Critic for discrete actions. It inherits from SAC for continuous actions and only changes a few
    methods."""
    agent_name = "SAC"

    def __init__(self, config, actor=None, actor_optimizer=None):
        Base_Agent.__init__(self, config)
        assert self.action_types == "DISCRETE", "Action types must be discrete. Use SAC instead for continuous actions"
        assert self.config.hyperparameters["Actor"][
                   "final_layer_activation"] == "Softmax", "Final actor layer must be softmax"
        self.hyperparameters = config.hyperparameters
        self.actor_local = actor
        self.prev_hid_state = None
        self.prev_prev_hid_state = None
        self.actor_optimizer = actor_optimizer
        self.opts = self.actor_local.opts

        # JI: reference encoder for critic
        # we do not use critic at inference time
        self.ref_enc = TextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=len(self.actor_local.trg_vocab),
            bidirectional=self.opts.model['enc_bidirectional'],
            rnn_type=self.opts.model['enc_type'],
            proj_dim=self.opts.model['enc_proj_dim'],
            proj_activ=self.opts.model['enc_proj_activ'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            layer_norm=self.opts.model['enc_lnorm'])

        self.ref_enc.to(DEVICE)

        # JI: reference encoder for critic
        # we do not use critic at inference time
        self.ref_enc1 = TextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=len(self.actor_local.trg_vocab),
            bidirectional=self.opts.model['enc_bidirectional'],
            rnn_type=self.opts.model['enc_type'],
            proj_dim=self.opts.model['enc_proj_dim'],
            proj_activ=self.opts.model['enc_proj_activ'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            layer_norm=self.opts.model['enc_lnorm'])

        self.ref_enc1.to(DEVICE)

        self.nll_loss = nn.NLLLoss(reduction='none', ignore_index=0)

        ctx_sizes = {str('src'): actor.enc.ctx_size}

        Decoder = get_decoder('cond')
        self.critic_local = Decoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.actor_local.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=ctx_sizes,
            ctx_name='src',
            tied_emb=self.opts.model['tied_emb'],
            att_type=self.opts.model['att_type'],
            att_temp=self.opts.model['att_temp'],
            att_activ=self.opts.model['att_activ'],
            att_ctx2hid=self.opts.model['att_ctx2hid'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            sched_sample=self.opts.model['sched_sampling'],
            out_logic=self.opts.model['out_logic'])

        self.critic_local.to(DEVICE)

        # self.discriminator = self.create_NN(self.state_size, self.num_skills, key_to_use="DISCRIMINATOR")
        self.critic_local_2 = Decoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.actor_local.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=ctx_sizes,
            ctx_name='src',
            tied_emb=self.opts.model['tied_emb'],
            att_type=self.opts.model['att_type'],
            att_temp=self.opts.model['att_temp'],
            att_activ=self.opts.model['att_activ'],
            att_ctx2hid=self.opts.model['att_ctx2hid'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            sched_sample=self.opts.model['sched_sampling'],
            out_logic=self.opts.model['out_logic'])

        self.critic_local_2.to(DEVICE)

        # JI: reference encoder for critic
        # we do not use critic at inference time
        self.ref_enc_target = TextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=len(self.actor_local.trg_vocab),
            bidirectional=self.opts.model['enc_bidirectional'],
            rnn_type=self.opts.model['enc_type'],
            proj_dim=self.opts.model['enc_proj_dim'],
            proj_activ=self.opts.model['enc_proj_activ'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            layer_norm=self.opts.model['enc_lnorm'])

        self.ref_enc_target.to(DEVICE)

        # JI: reference encoder for critic
        # we do not use critic at inference time
        self.ref_enc1_target = TextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=len(self.actor_local.trg_vocab),
            bidirectional=self.opts.model['enc_bidirectional'],
            rnn_type=self.opts.model['enc_type'],
            proj_dim=self.opts.model['enc_proj_dim'],
            proj_activ=self.opts.model['enc_proj_activ'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            layer_norm=self.opts.model['enc_lnorm'])

        self.ref_enc1_target.to(DEVICE)

        self.critic_target = Decoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.actor_local.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=ctx_sizes,
            ctx_name='src',
            tied_emb=self.opts.model['tied_emb'],
            att_type=self.opts.model['att_type'],
            att_temp=self.opts.model['att_temp'],
            att_activ=self.opts.model['att_activ'],
            att_ctx2hid=self.opts.model['att_ctx2hid'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            sched_sample=self.opts.model['sched_sampling'],
            out_logic=self.opts.model['out_logic'])

        self.critic_target.to(DEVICE)

        self.critic_target_2 = Decoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.actor_local.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=ctx_sizes,
            ctx_name='src',
            tied_emb=self.opts.model['tied_emb'],
            att_type=self.opts.model['att_type'],
            att_temp=self.opts.model['att_temp'],
            att_activ=self.opts.model['att_activ'],
            att_ctx2hid=self.opts.model['att_ctx2hid'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            sched_sample=self.opts.model['sched_sampling'],
            out_logic=self.opts.model['out_logic'])

        self.critic_target_2.to(DEVICE)
        params_critic = list(self.critic_local.parameters()) + list(self.ref_enc.parameters())
        params_critic2 = list(self.critic_local_2.parameters()) + list(self.ref_enc1.parameters())

        self.critic_optimizer = torch.optim.Adam(params_critic,
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(params_critic2,
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)

        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)

        Base_Agent.copy_model_over(self.ref_enc, self.ref_enc_target)
        Base_Agent.copy_model_over(self.ref_enc1, self.ref_enc1_target)

        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed)
        self.replay_mode = False
        self.state_dict_in = None

        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]
        assert not self.hyperparameters[
            "add_extra_noise"], "There is no add extra noise option for the discrete version of SAC at moment"
        self.add_extra_noise = False
        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]
        self.critic1_gl_loss = 0.0
        self.critic2_gl_loss = 0.0
        self.actor_gl_loss = 0.0
        self.gl_step_count = 1

    def produce_action_and_action_info(self, src_batch=None, batch=None, ref_batch=None, prev_hid_state=None,
                                       action=None):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""

        action = torch.LongTensor(action).to(DEVICE)

        if prev_hid_state == None:
            prev_hid_state = self.actor_local.dec.f_init(self.state_dict_in)

        prev_action = self.actor_local.dec.emb(action)
        log_action_probabilities, prev_hid_state, action_logits = self.actor_local.dec.f_next(self.state_dict_in,
                                                                                              prev_action,
                                                                                              prev_hid_state)
        action_probabilities = F.softmax(action_logits, dim=-1)

        max_probability_action = torch.argmax(action_probabilities, dim=-1)
        action_distribution = create_actor_distribution(self.action_types, action_probabilities, self.action_size)
        action = action_distribution.sample().cpu()
        return action, (
        action_probabilities, log_action_probabilities), max_probability_action, prev_hid_state

    def calculate_target_value(self, action_batch, reward_batch, ref_batch,
                                true_q_values, mask_batch, src_batch, batch=None, prev_hid_state=None, prev_q1=None, prev_q2=None):

        action_num = torch.LongTensor(action_batch)
        action_emb = self.actor_local.dec.emb(action_num.to(DEVICE))

        prev_prev_hid_state = prev_hid_state

        with torch.no_grad():
            next_state_action, (
            action_probabilities, log_action_probabilities), _, prev_hid_state = self.produce_action_and_action_info(
                src_batch=src_batch, batch=batch, ref_batch=ref_batch, action=action_batch,
                prev_hid_state=prev_prev_hid_state)
            new_emb = action_emb.detach()

            if prev_q1 == None:
                prev_q1 = self.critic_target.f_init(batch)
                prev_q2 = self.critic_target_2.f_init(batch)

            _, prev_q1, qf1_next_target = self.critic_target.f_next(ref_batch[0], new_emb, prev_q1)
            _, prev_q2, qf2_next_target = self.critic_target_2.f_next(ref_batch[1], new_emb, prev_q2)

            local_min = true_q_values
            # JI: uncomment this for supervised reward
            # local_min = torch.min(qf1_next_target, qf2_next_target)

            min_qf_next_target = action_probabilities * (local_min - self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.mean(dim=1).unsqueeze(-1)

            next_q_value = reward_batch + (1.0 - mask_batch.unsqueeze(-1)) * self.hyperparameters["discount_rate"] * (
                min_qf_next_target)

        return next_q_value, prev_hid_state, prev_q1, prev_q2

    def calculate_critic_losses(self, ref_batch, next_q_value, action_batch, mask_batch, prev_action=None,
                                 prev_q1=None, prev_q2=None, batch=None):

        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
                term is taken into account"""

        action_num = torch.LongTensor(action_batch)

        prev_action_num = torch.LongTensor(prev_action)
        prev_action_emb = self.actor_local.dec.emb(prev_action_num.to(DEVICE))
        new_emb = prev_action_emb.detach()

        if prev_q1 == None:
            prev_q1 = self.critic_local.f_init(batch)
            prev_q2 = self.critic_local_2.f_init(batch)

        _, prev_q1, qf1 = self.critic_local.f_next(ref_batch[0], new_emb, prev_q1)
        _, prev_q2, qf2 = self.critic_local_2.f_next(ref_batch[1], new_emb, prev_q2)

        qf1 = qf1.gather(1, action_num.to(DEVICE).unsqueeze(-1))
        qf2 = qf2.gather(1, action_num.to(DEVICE).unsqueeze(-1))

        qf1_loss = F.mse_loss(qf1.squeeze(-1) * (1.0 - mask_batch),
                              next_q_value.squeeze(-1).to(DEVICE) * (1.0 - mask_batch), reduction='none')
        qf2_loss = F.mse_loss(qf2.squeeze(-1) * (1.0 - mask_batch),
                              next_q_value.squeeze(-1).to(DEVICE) * (1.0 - mask_batch), reduction='none')

        return qf1_loss, qf2_loss, prev_q1, prev_q2

    def calculate_actor_loss(self, ref_batch, src_batch=None, batch=None, true_q_values=None,
                             prev_hid_state=None, prev_action=None, prev_q2=None, prev_q1=None, mask=None):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        prev_prev_hid_state = prev_hid_state
        action, (action_probabilities,
                 log_action_probabilities), max_act, prev_hid_state = self.produce_action_and_action_info(
            src_batch=src_batch, batch=batch, ref_batch=None, action=prev_action,
            prev_hid_state=prev_prev_hid_state)
        prep_state_batch_num = torch.LongTensor(prev_action)
        prep_state_batch_emb = self.actor_local.dec.emb(prep_state_batch_num.to(DEVICE))

        if prev_prev_hid_state == None:
            prev_prev_hid_state = self.actor_local.dec.f_init(self.state_dict_in)

        new_emb = prep_state_batch_emb.detach()

        if prev_q1 == None:
            prev_q1 = self.critic_local.f_init(batch)
            prev_q2 = self.critic_local_2.f_init(batch)

        _, prev_q1, qf1_pi = self.critic_local.f_next(ref_batch[0], new_emb, prev_q1.detach())
        _, prev_q2, qf2_pi = self.critic_local_2.f_next(ref_batch[1], new_emb, prev_q2.detach())

        # JI: uncomment this for supervised reward
        # min_qf_pi = torch.min(qf1_pi, qf2_pi)
        min_qf_pi = true_q_values.to(DEVICE)

        inside_term = (self.alpha * log_action_probabilities / (
                    1e-8 + log_action_probabilities.norm(p=2, dim=1, keepdim=True))) - min_qf_pi

        mask = (1.0 - mask.unsqueeze(-1)).repeat(1, action_probabilities.size()[1])
        policy_loss = action_probabilities * inside_term * mask

        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities, prev_hid_state, prev_q1, prev_q2
