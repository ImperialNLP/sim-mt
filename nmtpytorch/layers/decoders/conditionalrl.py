from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F

from ...utils.nn import get_rnn_hidden_state, get_activation_fn
from .. import FF
from ..attention import get_attention


class ConditionalDecoderRL(nn.Module):
    """A conditional decoder with attention à la dl4mt-tutorial."""
    def __init__(self, input_size, hidden_size, ctx_size_dict, ctx_name, n_vocab,
                 rnn_type, tied_emb=False, dec_init='zero', dec_init_activ='tanh',
                 dec_init_size=None, att_type='mlp',
                 att_activ='tanh', att_bottleneck='ctx', att_temp=1.0,
                 att_ctx2hid=True,
                 transform_ctx=True, mlp_bias=False, dropout_out=0,
                 emb_maxnorm=None, emb_gradscale=False, sched_sample=0,
                 out_logic='simple', dec_inp_activ=None, critic=False):
        super().__init__()

        self.critic = critic

        # Normalize case
        self.rnn_type = rnn_type.upper()
        self.out_logic = out_logic

        # A persistent dictionary to save activations for further debugging
        # Currently only used in MMT decoder
        self.persistence = defaultdict(list)

        # Safety checks
        assert self.rnn_type in ('GRU', 'LSTM'), \
            "rnn_type '{}' not known".format(rnn_type)
        assert dec_init.startswith(('zero', 'feats', 'sum_ctx', 'mean_ctx', 'max_ctx', 'last_ctx')), \
            "dec_init '{}' not known".format(dec_init)

        RNN = getattr(nn, '{}Cell'.format(self.rnn_type))
        # LSTMs have also the cell state
        self.n_states = 1 if self.rnn_type == 'GRU' else 2

        # Set custom handlers for GRU/LSTM
        if self.rnn_type == 'GRU':
            self._rnn_unpack_states = lambda x: x
            self._rnn_pack_states = lambda x: x
        elif self.rnn_type == 'LSTM':
            self._rnn_unpack_states = self._lstm_unpack_states
            self._rnn_pack_states = self._lstm_pack_states

        # Set decoder initializer
        self._init_func = getattr(self, '_rnn_init_{}'.format(dec_init))

        # Other arguments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size_dict = ctx_size_dict
        self.ctx_name = ctx_name
        self.n_vocab = n_vocab
        self.tied_emb = tied_emb
        self.dec_init = dec_init
        self.dec_init_size = dec_init_size
        self.dec_init_activ = dec_init_activ
        self.att_bottleneck = att_bottleneck
        self.att_activ = att_activ
        self.att_type = att_type
        self.att_temp = att_temp
        self.transform_ctx = transform_ctx
        self.att_ctx2hid = att_ctx2hid
        self.mlp_bias = mlp_bias
        self.dropout_out = dropout_out
        self.emb_maxnorm = emb_maxnorm
        self.emb_gradscale = emb_gradscale
        self.sched_sample = sched_sample
        self.dec_inp_activ_fn = get_activation_fn(dec_inp_activ)

        # Create target embeddings
        self.emb = nn.Embedding(self.n_vocab, self.input_size,
                                padding_idx=0, max_norm=self.emb_maxnorm,
                                scale_grad_by_freq=self.emb_gradscale)

        if self.att_type and self.att_type != 'uniform':
            # Create attention layer
            Attention = get_attention(self.att_type)
            self.att = Attention(
                self.ctx_size_dict[self.ctx_name],
                200,
                transform_ctx=self.transform_ctx,
                ctx2hid=self.att_ctx2hid,
                mlp_bias=self.mlp_bias,
                att_activ=self.att_activ,
                att_bottleneck=self.att_bottleneck,
                temp=self.att_temp)

        if self.att_type and self.att_type == 'uniform':
            # Create attention layer
            Attention = get_attention(self.att_type)
            self.att = Attention()

        if self.dec_init != 'zero':
            # For source-based inits, input size is the encoding size
            # For 'feats', it's given by dec_init_size, no need to infer
            if self.dec_init.endswith('_ctx'):
                self.dec_init_size = self.ctx_size_dict[self.ctx_name]
            # Add a FF layer for decoder initialization
            self.ff_dec_init = FF(
                self.dec_init_size,
                self.hidden_size * self.n_states,
                activ=self.dec_init_activ)

        # Create decoders
        self.dec0 = RNN(self.input_size, self.hidden_size)
        if self.att_type:
            # If no attention, do not add the 2nd GRU
            self.dec1 = RNN(self.hidden_size, self.hidden_size)

        # Output dropout
        if self.dropout_out > 0:
            self.do_out = nn.Dropout(p=self.dropout_out)

        # Output bottleneck: maps hidden states to target emb dim
        # simple: tanh(W*h)
        #   deep: tanh(W*h + U*emb + V*ctx)
        out_inp_size = self.hidden_size

        # Dummy op to return back the hidden state for simple output
        self.out_merge_fn = lambda h, c: h

        #if self.out_logic == 'deep':
        #    out_inp_size += self.hidden_size + 2
        #    self.out_merge_fn = lambda h, e, c: torch.cat((h, e, c), dim=1)
        #    #out_inp_size += self.hidden_size
        #    #self.out_merge_fn = lambda h, c: torch.cat((h, c), dim=1)
        #    #print(out_inp_size)
        #    #exit()

        #if self.critic:
        #    out_inp_size = self.input_size

        # Final transformation that receives concatenated outputs or only h
        self.hid2out = FF(out_inp_size, self.input_size,
                          bias_zero=True, activ='tanh')

        # Final softmax
        self.out2prob = FF(self.input_size, self.n_vocab)

        # Tie input embedding matrix and output embedding matrix
        if self.tied_emb:
            self.out2prob.weight = self.emb.weight

        self.lin_class = nn.Linear(1, 1, bias=False)
        self.nll_loss = nn.NLLLoss(reduction="sum", ignore_index=0)

    def _lstm_pack_states(self, h):
        """Pack LSTM hidden and cell state."""
        return torch.cat(h, dim=-1)

    def _lstm_unpack_states(self, h):
        """Unpack LSTM hidden and cell state to tuple."""
        return torch.split(h, self.hidden_size, dim=-1)

    def _rnn_init_zero(self, ctx_dict):
        """Zero initialization."""
        ctx, _ = ctx_dict[self.ctx_name]
        return torch.zeros(
            ctx.shape[1], self.hidden_size * self.n_states, device=ctx.device)

    def _rnn_init_mean_ctx(self, ctx_dict):
        """Initialization with mean-pooled source annotations."""
        ctx, ctx_mask = ctx_dict[self.ctx_name]
        return self.ff_dec_init(
            ctx.sum(0).div(ctx_mask.unsqueeze(-1).sum(0))
            if ctx_mask is not None else ctx.mean(0))

    def _rnn_init_sum_ctx(self, ctx_dict):
        """Initialization with sum-pooled source annotations."""
        ctx, ctx_mask = ctx_dict[self.ctx_name]
        assert ctx_mask is None
        return self.ff_dec_init(ctx.sum(0))

    def _rnn_init_max_ctx(self, ctx_dict):
        """Initialization with max-pooled source annotations."""
        ctx, ctx_mask = ctx_dict[self.ctx_name]
        # Max-pooling may not care about mask (depends on non-linearity maybe)
        return self.ff_dec_init(ctx.max(0)[0])

    def _rnn_init_last_ctx(self, ctx_dict):
        """Initialization with the last source annotation."""
        ctx, ctx_mask = ctx_dict[self.ctx_name]
        if ctx_mask is None:
            h_0 = self.ff_dec_init(ctx[-1])
        else:
            last_idxs = ctx_mask.sum(0).sub(1).long()
            h_0 = self.ff_dec_init(ctx[last_idxs, range(ctx.shape[1])])
        return h_0

    def _rnn_init_feats(self, ctx_dict):
        """Feature based decoder initialization."""
        return self.ff_dec_init(ctx_dict['feats'][0].squeeze(0))

    def get_emb(self, idxs, tstep=-1):
        """Returns time-step based embeddings."""
        return self.emb(idxs)

    def f_init(self, ctx_dict):
        """Returns the initial h_0 for the decoder."""
        self.history = defaultdict(list)
        return self._init_func(ctx_dict)

    def f_next(self, ctx_dict, y, h, emb):
        """Applies one timestep of recurrence."""
        img_z_t = None
        if self.att_type and not self.critic:
            # Apply attention
            img_alpha_t, img_z_t = self.att(
                emb.unsqueeze(0), *ctx_dict[self.ctx_name])

            if not self.training:
                self.history['alpha_img'].append(img_alpha_t)

            y = torch.cat((y, img_z_t), 1)

        h1_c1 = self.dec0(y, self._rnn_unpack_states(h))
        h1 = get_rnn_hidden_state(h1_c1)
        h_return = self._rnn_pack_states(h1_c1)
        final_hid_in = h1

        # Output logic
        logit = self.hid2out(final_hid_in)

        # Apply dropout if any
        if self.dropout_out > 0:
            logit = self.do_out(logit)

        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim
        logit = self.out2prob(logit)

        # Return log probs and new hidden states
        return logit, h_return, img_z_t

    def forward(self, h, policy_input, ctx_dict, emb, sample=True):
        """Computes the softmax outputs given source annotations `ctx_dict[self.ctx_name]`
        and ground-truth target token indices `y`. Only called during training.

        Arguments:
            ctx_dict(dict): A dictionary of tensors that should at least contain
                the key `ctx_name` as the main source representation of shape
                S*B*ctx_dim`.
            y(Tensor): A tensor of `T*B` containing ground-truth target
                token indices for the given batch.
        """

        loss = 0.0

        if self.n_vocab == 1:
            logit, h, img_ctx = self.f_next(ctx_dict, policy_input, h, emb)
            log_p = self.lin_class(logit).squeeze(-1)
            action = torch.round(log_p).squeeze(-1)
            action_all = action
            act_dist = torch.distributions.Categorical(
                probs=torch.cat((log_p, 1 - log_p), dim=-1))
            ent = act_dist.entropy()
        else:
            logit, h, img_ctx = self.f_next(ctx_dict, policy_input, h, emb)
            p_dist = F.softmax(logit, dim=-1)
            if sample:
                p_dist_gumbel = F.gumbel_softmax(logit, tau=1, hard=True)
                action_all = p_dist_gumbel
                action = p_dist_gumbel[:, 1]
                log_p = (p_dist * p_dist_gumbel).sum(-1)
                ent = -(p_dist * torch.log(p_dist)).sum(-1)
            else:
                act_dist = torch.distributions.Categorical(probs=p_dist)
                action = p_dist.argmax(dim=-1)
                action_all = F.one_hot(action, num_classes=2)
                log_p = act_dist.log_prob(action)
                ent = act_dist.entropy()

        return {
            'loss': loss,
            'action': action,
            'log_p': log_p,
            'h': h, 'negentropy': -ent,
            'action_all': action_all,
            'img_ctx': img_ctx,
        }
