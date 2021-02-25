import logging

import torch
from torch import nn

import numpy as np

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from ..layers import VisualFeaturesEncoder
from ..layers.decoders import get_decoder
from ..utils.misc import get_n_params
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..utils.data import sort_predictions
from ..datasets import MultimodalDataset
from ..metrics import Metric
from ..utils.misc import load_pt_file
from ..config import Options
from .. import models
from ..layers import FF

# modifs to main code by Julia Ive
# julia.ive84@gmail.com


logger = logging.getLogger('nmtpytorch')


class SIMRL(nn.Module):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'emb_dim': 128,             # Source and target embedding sizes
            'emb_maxnorm': None,        # Normalize embeddings l2 norm to 1
            'emb_gradscale': False,     # Scale embedding gradients w.r.t. batch frequency
            'enc_dim': 256,             # Encoder hidden size
            'enc_type': 'gru',          # Encoder type (gru|lstm)
            'enc_lnorm': False,         # Add layer-normalization to encoder output
            'enc_bidirectional': True,  # Whether the RNN encoder should be bidirectional
            'n_encoders': 1,            # Number of stacked encoders
            'dec_dim': 256,             # Decoder hidden size
            'dec_type': 'gru',          # Decoder type (gru|lstm)
            'dec_variant': 'cond',      # (cond|simplegru|vector)
            'dec_init': 'mean_ctx',     # How to initialize decoder (zero/mean_ctx/feats)
            'dec_init_size': None,      # feature vector dimensionality for
            'dec_init_activ': 'tanh',   # Decoder initialization activation func
                                        # dec_init == 'feats'
            'att_type': 'mlp',          # Attention type (mlp|dot)
            'att_temp': 1.,             # Attention temperature
            'att_activ': 'tanh',        # Attention non-linearity (all torch nonlins)
            'att_mlp_bias': False,      # Enables bias in attention mechanism
            'att_bottleneck': 'ctx',    # Bottleneck dimensionality (ctx|hid)
            'att_transform_ctx': True,  # Transform annotations before attention
            'att_ctx2hid': True,        # Add one last FC layer on top of the ctx
            'dropout_emb': 0,           # Simple dropout to source embeddings
            'dropout_ctx': 0,           # Simple dropout to source encodings
            'dropout_out': 0,           # Simple dropout to decoder output
            'dropout_enc': 0,           # Intra-encoder dropout if n_encoders > 1
            'tied_emb': False,          # Share embeddings: (False|2way|3way)
            'direction': None,          # Network directionality, i.e. en->de
            'max_len': 80,              # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       # Curriculum: ascending/descending/None
            'sampler_type': 'bucket',   # bucket or approximate
            'sched_sampling': 0,        # Scheduled sampling ratio
            'short_list': 0,            # Short list vocabularies (0: disabled)
            'bos_type': 'emb',          # 'emb': default learned emb
            'bos_activ': None,          #
            'bos_dim': None,            #
            'out_logic': 'simple',      # 'simple' or 'deep' output
            'dec_inp_activ': None,      # Non-linearity for GRU2 input in dec
            'act_count': 2,
            'translator_type': 'srlgs',
            'translator_args': {},      # No extra arguments to translator
            'feat_mode': None,          # May be used for more sophisticated aux features
            'mm_agent_init': False,
            'mm_agent_att': False,
            'mm_env': False,
            'aux_dim': None,            # Auxiliary features dim (# channels for conv features)
            'aux_dim_1': None,          # conv filters
            'aux_dropout': 0.0,         # Auxiliary features dropout
            'aux_lnorm': False,         # layer-norm
            'aux_l2norm': False,        # L2-normalize
            'aux_proj_dim': None,       # Projection layer for features
            'aux_proj_activ': None,     # Projection layer non-linearity
            'max_cw': 5,
            'target_delay': 5,
            'alpha': 1,
            'beta': 1,
            'ent_penalty': 1,
            'env_file': None
        }

    def __init__(self, opts):
        super().__init__()

        # opts -> config file sections {.model, .data, .vocabulary, .train}
        self.batch_size = 32
        self.opts = opts

        # Vocabulary objects
        self.vocabs = {}

        # Each auxiliary loss should be stored inside this dictionary
        # in order to be taken into account by the mainloop for multi-tasking
        self.aux_loss = {}

        # Setup options
        self.opts.model = self.set_model_options(opts.model)
        self.opts.model['translator_args'] = {'k': 'hhh'}

        self.critic_loss = nn.MSELoss(reduction='none')

        # Parse topology & languages
        self.topology = Topology(self.opts.model['direction'])

        # Load vocabularies here
        for name, fname in self.opts.vocabulary.items():
            self.vocabs[name] = Vocabulary(fname, short_list=self.opts.model['short_list'])

        # Inherently non multi-lingual aware
        slangs = self.topology.get_src_langs()
        tlangs = self.topology.get_trg_langs()
        if slangs:
            self.sl = slangs[0]
            self.src_vocab = self.vocabs[self.sl]
            self.n_src_vocab = len(self.src_vocab)
        if tlangs:
            self.tl = tlangs[0]
            self.trg_vocab = self.vocabs[self.tl]
            self.n_trg_vocab = len(self.trg_vocab)
            # Need to be set for early-stop evaluation
            # NOTE: This should come from config or elsewhere
            self.val_refs = self.opts.data['val_set'][self.tl]

        # Check vocabulary sizes for 3way tying
        if self.opts.model.get('tied_emb', False) not in [False, '2way', '3way']:
            raise RuntimeError(
                "'{}' not recognized for tied_emb.".format(self.opts.model['tied_emb']))

        if self.opts.model.get('tied_emb', False) == '3way':
            assert self.n_src_vocab == self.n_trg_vocab, \
                "The vocabulary sizes do not match for 3way tied embeddings."

    def __repr__(self):
        s = super().__repr__() + '\n'
        for vocab in self.vocabs.values():
            s += "{}\n".format(vocab)
        s += "{}\n".format(get_n_params(self))
        return s

    def set_model_options(self, model_opts):
        self.set_defaults()
        for opt, value in model_opts.items():
            if opt in self.defaults:
                # Override defaults from config
                self.defaults[opt] = value
            else:
                logger.info('Warning: unused model option: {}'.format(opt))
        return self.defaults

    def cache_enc_states(self, batch):
        """Caches encoder states internally by forward-pass'ing each decoder."""
        for key, enc in self.encoders.items():
            _ = enc(batch[key])

    def get_enc_state_dict(self, up_to=int(1e6)):
        """Encodes the batch optionally by partial encoding up to `up_to`
        words for derived simultaneous NMT classes. By default, the value
        is large enough to leave it as vanilla NMT."""
        return {str(k): e.get_states(up_to=up_to) for k, e in self.encoders.items()}

    def init_trans_model(self):
        data = load_pt_file(self.opts.model['env_file'])
        weights, _, opts = data['model'], data['history'], data['opts']
        opts = Options.from_dict(opts)
        instance = getattr(models, opts.train['model_type'])(opts=opts)

        # Setup layers
        instance.setup(is_train=False)

        # Load weights
        instance.load_state_dict(weights, strict=False)

        # Move to device
        instance.to(DEVICE)

        # Switch to eval mode
        instance.train(False)

        # JI: assign environment
        self.model_trans = instance

    def reset_parameters(self):
        for name, param in self.named_parameters():
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)
        # Reset padding embedding to 0
        if hasattr(self, 'enc') and hasattr(self.enc, 'emb'):
            with torch.no_grad():
                self.enc.emb.weight.data[0].fill_(0)

        self.init_trans_model()

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""
        ########################
        # Create Textual Encoder
        ########################
        self.emb = nn.Embedding(self.n_trg_vocab, self.opts.model['emb_dim'],
                                padding_idx=0, max_norm=self.opts.model['emb_maxnorm'],
                                scale_grad_by_freq=self.opts.model['emb_gradscale'])

        self.ctx_sizes = {'image': self.opts.model['aux_proj_dim']}

        ################
        # Create Decoder
        ################
        # JI; here is the modified decoder (to return log probs) that we use as an actor
        Decoder = get_decoder('condrl')
        if self.opts.model['mm_agent_att']:
            att_type = self.opts.model['att_type']
            ctx_name = 'image'
            dec_in_size = 2 + 2 * self.opts.model['emb_dim'] + self.opts.model['enc_dim']
        else:
            att_type = None
            ctx_name = str(self.sl)
            dec_in_size = 2 + self.opts.model['emb_dim'] + self.opts.model['enc_dim']

        self.enc = Decoder(
            input_size=dec_in_size,
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=1,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=ctx_name,
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            dec_init_size=self.opts.model['dec_init_size'],
            dec_init_activ=self.opts.model['dec_init_activ'],
            att_type=att_type,
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
            out_logic=self.opts.model['out_logic'],
            dec_inp_activ=self.opts.model['dec_inp_activ'],
            critic=True)

        self.dec = Decoder(
            input_size=dec_in_size,
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=2,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=ctx_name,
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            dec_init_size=self.opts.model['dec_init_size'],
            dec_init_activ=self.opts.model['dec_init_activ'],
            att_type=att_type,
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
            out_logic=self.opts.model['out_logic'],
            dec_inp_activ=self.opts.model['dec_inp_activ'])

        if self.opts.model['mm_agent_init']:
            self.imgctx2hid = FF(self.opts.model['aux_dim_1'] * self.opts.model['aux_dim'], self.opts.model['aux_proj_dim'])

        if self.opts.model['mm_agent_att']:
            encoders = {}
            encoders['image'] = VisualFeaturesEncoder(
                input_size=self.opts.model['aux_dim'],
                proj_dim=self.opts.model['aux_proj_dim'],
                proj_activ=self.opts.model['aux_proj_activ'],
                layer_norm=self.opts.model['aux_lnorm'],
                l2_norm=self.opts.model['aux_l2norm'],
                dropout=self.opts.model['aux_dropout'])
            self.encoders = nn.ModuleDict(encoders)

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.enc.emb.weight = self.dec.emb.weight

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        self.dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model['max_len'],
            bucket_order=self.opts.model['bucket_order'],
            sampler_type=self.opts.model['sampler_type'])
        logger.info(self.dataset)
        return self.dataset

    def get_bos(self, batch_size):
        """Returns a representation for <bos> embeddings for decoding."""
        return torch.LongTensor(batch_size).fill_(self.trg_vocab['<bos>'])

    def encode(self, batch, **kwargs):
        """Encodes all inputs and returns a dictionary.

        Arguments:
            batch (dict): A batch of samples with keys designating the
                information sources.

        Returns:
            dict:
                A dictionary where keys are source modalities compatible
                with the data loader and the values are tuples where the
                elements are encodings and masks. The mask can be ``None``
                if the relevant modality does not require a mask.
        """
        d = {str(self.sl): self.enc(batch[self.sl])}
        if 'feats' in batch:
            d['feats'] = (batch['feats'], None)
        return d

    def forward(self, batch, **kwargs):
        """Computes the forward-pass of the network and returns batch loss.

        Arguments:
            batch (dict): A batch of samples with keys designating the source
                and target modalities.

        Returns:
            Tensor:
                A scalar loss normalized w.r.t batch size and token counts.
        """
        btl = batch[self.tl].repeat(1, 5)
        stl = batch[self.sl].repeat(1, 5)
        if self.opts.model['mm_agent_init'] or self.opts.model['mm_agent_att'] or self.opts.model['mm_env']:
            img = batch['image'].repeat(1, 5, 1)

        batch[self.tl] = btl
        batch[self.sl] = stl
        if self.opts.model['mm_agent_init'] or self.opts.model['mm_agent_att'] or self.opts.model['mm_env']:
            batch['image'] = img

        batch_length = batch.size * 5

        # JI: our outputs
        actions = []
        log_probs_list = []
        neg_ents_list = []
        wait_length = []
        estimated_values = None

        wait_length_counter = [0] * batch_length

        # JI: arrays to store read counters and boolean to seq end
        src_read_counter = [1] * batch_length
        end_of_seq = [False] * batch_length

        src_max_len = batch[self.sl].size(0)
        vocab = self.model_trans.trg_vocab

        eos = vocab['<eos>']
        unk = vocab['<unk>']

        # JI: max decoding length
        max_len = 40
        translations = np.full((batch_length, max_len), eos)
        translations_mask = np.full((batch_length, max_len), 0)

        batch.device(DEVICE)

        self.model_trans.cache_enc_states(batch)

        # JI: take the first state for the first action
        with torch.no_grad():
            ctx_dict = self.model_trans.get_enc_state_dict()

        state_dict_full = ctx_dict[self.sl][0]
        state_dict_prep = state_dict_full[0].unsqueeze(0)

        state_dict = state_dict_prep.clone()
        state_dict_mask = torch.ones(1, batch_length).to(DEVICE)

        # JI: Compute decoder's initial state h_0 for each sentence (BxE)
        # apparently None for simultaneos decoder ???
        prev_h = self.model_trans.dec.f_init(ctx_dict)

        # Start all sentences with <s>
        prev_word_ind = self.model_trans.get_bos(batch_length).to(DEVICE)
        with torch.no_grad():
            prev_word = self.model_trans.dec.emb(prev_word_ind)

        action = torch.zeros(batch_length, device=DEVICE)
        action_all = torch.zeros([batch_length, 2], device=DEVICE)

        if self.opts.model['mm_agent_init']:
            h_0 = self.imgctx2hid(batch['image'].permute(1, 0, 2).reshape(batch_length, -1))
            h_policy = h_0
            h_critic = h_0
        else:
            h_policy = torch.zeros(batch_length, self.opts.model['dec_dim'], device=DEVICE)
            h_critic = torch.zeros(batch_length, self.opts.model['dec_dim'], device=DEVICE)

        if self.opts.model['mm_agent_att']:
            self.cache_enc_states(batch)
            # get image encoded state_dicts for RL agent
            image_ctx_vec = self.get_enc_state_dict()
        else:
            image_ctx_vec = None

        trg_step = 0

        # JI: loop until we generate <eos> or hit max
        while trg_step < max_len:
            if self.opts.model['mm_env']:
                #JI: get next word
                state_dict_in = {'src': (state_dict, state_dict_mask), 'image': ctx_dict['image']}
            else:
                state_dict_in = {'src': (state_dict, state_dict_mask)}
            #logp, new_h, new_word, ctx_vec = self.stranslator.decoder_step(state_dict_in, prev_word, prev_h)
            with torch.no_grad():
                logp, new_h, ctx_vec = self.model_trans.dec.f_next(state_dict_in, prev_word, prev_h)
                new_word_ind = logp.argmax(1)
                new_word = self.model_trans.dec.emb(new_word_ind)

            #JI: concat before input to actor
            policy_input = torch.cat((action_all.float(), new_word.detach(), ctx_vec.detach()), 1)

            #JI: get actor output
            out_dict = self.dec(h_policy, policy_input, image_ctx_vec, new_word.detach())

            action_all = out_dict['action_all']
            img_ctx_weighted = out_dict['img_ctx']
            with torch.no_grad():
                if self.opts.model['mm_agent_att']:
                    critic_input = torch.cat((action_all.float(), new_word.detach(), ctx_vec.detach(), img_ctx_weighted.detach()), 1)
                else:
                    critic_input = torch.cat((action_all.float(), new_word.detach(), ctx_vec.detach()), 1)

            estimate = self.enc(h_critic, critic_input, image_ctx_vec, new_word.detach())
            if estimated_values is not None:
                estimated_values = torch.cat((estimated_values, estimate['log_p'].unsqueeze(-1)), dim=-1)
            else:
                estimated_values = estimate['log_p'].unsqueeze(-1)

            h_policy = out_dict['h']
            h_critic = estimate['h']
            action = out_dict['action']
            log_p = out_dict['log_p']
            neg_ent = out_dict['negentropy']

            state_dict_new = torch.zeros(
                1, batch_length, self.opts.model['enc_dim']).to(DEVICE)
            state_dict_mask_new = torch.zeros(1, batch_length).to(DEVICE)

            #JI: we augment the dict size to accomodate new encoder states if our read counter tell us so
            if state_dict.size()[0] <= max(src_read_counter):
                state_dict = torch.cat((state_dict, state_dict_new), 0)
                state_dict_mask = torch.cat((state_dict_mask, state_dict_mask_new), 0)

            actions.append(action)
            # JI: get clones for inplace operations
            prev_word_cl = prev_word.clone()
            if prev_h is None:
                prev_h = new_h

            prev_h_cl = prev_h.clone()
            state_dict_cl = state_dict.clone()
            state_dict_mask_cl = state_dict_mask.clone()

            # JI: loop over the batch samples
            batch_wait_length = []
            for n in range(batch_length):

                action_sent = action[n]

                translations_mask[n][trg_step] = 1
                if src_read_counter[n] >= src_max_len or end_of_seq[n]:
                    translations_mask[n][trg_step] = 0

                # JI: if we took all the encoder states or actition is WRITE
                if src_read_counter[n] >= src_max_len or action_sent == 1:
                    # JI: we update prev word
                    prev_word_cl[n] = new_word[n]

                    # JI: we add our new word
                    translations[n, trg_step] = new_word_ind[n].item()

                    prev_h_cl[n] = new_h[n]

                    # JI: we increment you wait counter
                    batch_wait_length.append(wait_length_counter[n])
                    wait_length_counter[n] = 0

                    if new_word_ind[n] == eos:
                        end_of_seq[n] = True

                else:
                    # JI: otherwise we read next word
                    state_dict_cl[src_read_counter[n]][n] = state_dict_full[src_read_counter[n]][n]
                    state_dict_mask_cl[-1][n] = 1
                    src_read_counter[n] += 1
                    wait_length_counter[n] += 1
                    # JI: temporary solution for now
                    # JI: we add UNK to the hypothesis
                    translations[n, trg_step] = unk

            wait_length.append(batch_wait_length)
            log_probs_list.append(log_p)
            neg_ents_list.append(neg_ent)

            # JI: update initial values

            prev_h = prev_h_cl
            prev_word = prev_word_cl
            state_dict = state_dict_cl
            state_dict_mask = state_dict_mask_cl
            trg_step += 1

        actions = torch.stack(actions, dim=0)
        log_probs_list = torch.stack(log_probs_list, dim=0).to(DEVICE)
        neg_ents_list = torch.stack(neg_ents_list, dim=0).to(DEVICE)

        #JI: compute BLEU reward
        #JI: we use a smoothing since BLEU is on the sent level
        chencherry = SmoothingFunction()
        rewards_list = []
        ref = batch[self.tl]
        seq_len = actions.size()[0]

        #JI: here comes the mask to not compute losses
        for n, line in enumerate(translations):
            ref_line = ref[:, n].cpu().tolist()

            line_good = self.model_trans.trg_vocab.idxs_to_sent(line)
            line = self.model_trans.trg_vocab.idxs_to_sent(line, debug=True)
            ref_line = self.model_trans.trg_vocab.idxs_to_sent(ref_line)

            line_ar = line.split()
            ref_ar = ref_line.split()
            line_good_ar = line_good.split()

            new_fake = line_ar
            new_real = ref_ar[1:]

            # JI: fill our empty hypothesis with UNKs for BLEU not to be affected by the hypothesis length
            new_fake2 = np.full((len(new_fake),), '<unk>', dtype='object')
            max_cw = self.opts.model['max_cw']
            target_delay = self.opts.model['target_delay']

            prev_bleu = 0.0
            scores_local = np.zeros(seq_len)
            # JI: add at each time step a new hypothesis word
            # JI: we compute the increment of BLEU (according to Gu paper)
            eos_flag = False
            delay_counter = 1
            new_hyp_counter = 0
            _src = 1
            _trg = 0
            _sum = 0
            for g, ww in enumerate(new_fake):
                if not new_fake[g] == '<unk>' and not eos_flag:
                    if delay_counter > 0:
                        delay_counter = 0

                    _trg += 1
                    _sum += _src
                    new_fake2[new_hyp_counter] = new_fake[g]

                    new_hyp_counter += 1
                    if new_fake[g] == '<eos>' or translations_mask[n][g] == 0:
                        local_score = sentence_bleu([new_real], line_good_ar, smoothing_function=chencherry.method5)
                        delay_rew = _sum / (_src * _trg + 1e-6)
                        delay_rew_fin = self.opts.model['beta'] * -np.maximum(delay_rew - target_delay, 0)
                        scores_local[g] = local_score + delay_rew_fin
                        eos_flag = True
                    else:
                        local_score = sentence_bleu([new_real], new_fake2.tolist(), smoothing_function=chencherry.method5)
                        scores_local[g] = local_score - prev_bleu
                        prev_bleu = local_score
                elif eos_flag:
                    scores_local[g] = 0.0

                elif new_fake[g] == '<unk>' and not eos_flag:
                    if _src < src_max_len:
                        _src += 1

                    delay_counter += 1
                    if delay_counter > max_cw:
                        scores_local[g] = self.opts.model['alpha'] * (delay_counter - max_cw)
                    else:
                        scores_local[g] = 0.0

            rewards_list.append(np.asarray(scores_local))

        rewards_list = np.transpose(np.stack(rewards_list, axis=0))
        # JI: computing cumulative reward
        gamma = 1
        cumulative_rewards = []

        for t in range(max_len):
            cum_value = np.zeros(batch_length)
            for s in range(t, seq_len):
                pw = np.power(gamma, (s - t))
                cum_value += pw * rewards_list[s]

            cumulative_rewards.append(cum_value)

        cumulative_rewards = torch.FloatTensor(cumulative_rewards).permute(1, 0).to(DEVICE)
        translations = torch.LongTensor(translations).to(DEVICE).permute(1, 0)
        translations_mask = torch.LongTensor(translations_mask).to(DEVICE).permute(1, 0)
        critic_loss = self.critic_loss(
            estimated_values * translations_mask.permute(1, 0),
            cumulative_rewards.to(DEVICE) * translations_mask.permute(1, 0))
        critic_loss = critic_loss.sum(-1).mean()

        # JI: baselines are coming from the critic's estimated state values.
        baselines = estimated_values.permute(1, 0)

        # JI: calculate the Advantages, A(s,a) = Q(s,a) - \hat{V}(s).
        loss = torch.zeros(batch_length,).to(DEVICE)
        ent_loss = torch.zeros(batch_length,).to(DEVICE)
        advantages = cumulative_rewards.permute(1, 0) - baselines

        reward_mean = torch.sum(translations_mask * advantages) / torch.sum(translations_mask)
        reward_mean2 = torch.sum(translations_mask * (advantages ** 2)) / torch.sum(translations_mask)
        reward_std = torch.sqrt(torch.max(reward_mean2 - reward_mean ** 2, torch.from_numpy(np.array([1e-7])).type(torch.float).to(DEVICE))) + 1e-7
        reward_c = advantages - reward_mean
        advantages = reward_c / reward_std

        # JI: use 0.1 for FR, ise 0.3 for DE
        ent_loss = torch.sum(torch.mul(neg_ents_list, translations_mask), dim=0).mean() * self.opts.model['ent_penalty']
        prep_loss = torch.sum(torch.mul(-log_probs_list, advantages.detach() * translations_mask), dim=0).mean()
        loss = prep_loss + ent_loss

        # JI: we min neg log prob
        result = {'loss': loss, 'n_items': 1}

        # JI: add critic loss as auxilliary
        self.aux_loss['critic_loss'] = critic_loss
        return result

    def test_performance(self, data_loader, dump_file=None):
        """Computes test set loss over the given DataLoader instance."""
        loss = Loss()

        for batch in pbar(data_loader, unit='batch'):
            batch.device(DEVICE)
            out = self.forward(batch)
            loss.update(out['loss'], out['n_items'])

        return [
            Metric('LOSS', loss.get(), higher_better=False),
        ]

    def register_tensorboard(self, handle):
        """Stores tensorboard hook for custom logging."""
        self.tboard = handle

    @staticmethod
    def beam_search(models, data_loader, beam_size=12, max_len=200,
                    lp_alpha=0., suppress_unk=False, n_best=False):
        """An efficient implementation for beam-search algorithm.

        Arguments:
            models (list of Model): Model instance(s) derived from `nn.Module`
                defining a set of methods. See `models/nmt.py`.
            data_loader (DataLoader): A ``DataLoader`` instance.
            beam_size (int, optional): The size of the beam. (Default: 12)
            max_len (int, optional): Maximum target length to stop beam-search
                if <eos> is still not generated. (Default: 200)
            lp_alpha (float, optional): If > 0, applies Google's length-penalty
                normalization instead of simple length normalization.
                lp: ((5 + |Y|)^lp_alpha / (5 + 1)^lp_alpha)
            suppress_unk (bool, optional): If `True`, suppresses the log-prob
                of <unk> token.
            n_best (bool, optional): If `True`, returns n-best list of the beam
                with the associated scores.

        Returns:
            list:
                A list of hypotheses in surface form.
        """
        def tile_ctx_dict(ctx_dict, idxs):
            """Returns dict of 3D tensors repeatedly indexed along the sample axis."""
            # 1st: tensor, 2nd optional mask
            return {
                k: (t[:, idxs], None if mask is None else mask[:, idxs])
                for k, (t, mask) in ctx_dict.items()
            }

        def check_context_ndims(ctx_dict):
            for name, (ctx, mask) in ctx_dict.items():
                assert ctx.dim() == 3, \
                    f"{name}'s 1st dim should always be a time dimension."

        # This is the batch-size requested by the user but with sorted
        # batches, efficient batch-size will be <= max_batch_size
        max_batch_size = data_loader.batch_sampler.batch_size
        k = beam_size
        inf = -1000
        results = []
        enc_args = {}

        decs = [m.dec for m in models]
        f_inits = [dec.f_init for dec in decs]
        f_nexts = [dec.f_next for dec in decs]
        vocab = models[0].trg_vocab

        # Common parts
        encoders = [m.encode for m in models]
        unk = vocab['<unk>']
        eos = vocab['<eos>']
        n_vocab = len(vocab)

        # Tensorized beam that will shrink and grow up to max_batch_size
        beam_storage = torch.zeros(
            max_len, max_batch_size, k, dtype=torch.long, device=DEVICE)
        mask = torch.arange(max_batch_size * k, device=DEVICE)
        nll_storage = torch.zeros(max_batch_size, device=DEVICE)

        for batch in pbar(data_loader, unit='batch'):
            batch.device(DEVICE)

            # Always use the initial storage
            beam = beam_storage.narrow(1, 0, batch.size).zero_()

            # Mask to apply to pdxs.view(-1) to fix indices
            nk_mask = mask.narrow(0, 0, batch.size * k)

            # nll: batch_size x 1 (will get expanded further)
            nll = nll_storage.narrow(0, 0, batch.size).unsqueeze(1)

            # Tile indices to use in the loop to expand first dim
            tile = range(batch.size)

            # Encode source modalities
            ctx_dicts = [encode(batch, **enc_args) for encode in encoders]

            # Sanity check one of the context dictionaries for dimensions
            check_context_ndims(ctx_dicts[0])

            # Get initial decoder state (N*H)
            h_ts = [f_init(ctx_dict) for f_init, ctx_dict in zip(f_inits, ctx_dicts)]

            # we always have <bos> tokens except that the returned embeddings
            # may differ from one model to another.
            idxs = models[0].get_bos(batch.size).to(DEVICE)

            for tstep in range(max_len):
                # Select correct positions from source context
                ctx_dicts = [tile_ctx_dict(cd, tile) for cd in ctx_dicts]

                # Get log probabilities and next state
                # log_p: batch_size x vocab_size (t = 0)
                #        batch_size*beam_size x vocab_size (t > 0)
                # NOTE: get_emb does not exist in some models, fix this.
                log_ps, h_ts = zip(
                    *[f_next(cd, dec.get_emb(idxs, tstep), h_t[tile]) for
                      f_next, dec, cd, h_t in zip(f_nexts, decs, ctx_dicts, h_ts)])

                # Do the actual averaging of log-probabilities
                log_p = sum(log_ps).data

                if suppress_unk:
                    log_p[:, unk] = inf

                # Detect <eos>'d hyps
                idxs = (idxs == 2).nonzero()
                if idxs.numel():
                    if idxs.numel() == batch.size * k:
                        break
                    idxs.squeeze_(-1)
                    # Unfavor all candidates
                    log_p.index_fill_(0, idxs, inf)
                    # Favor <eos> so that it gets selected
                    log_p.view(-1).index_fill_(0, idxs * n_vocab + 2, 0)

                # Expand to 3D, cross-sum scores and reduce back to 2D
                # log_p: batch_size x vocab_size ( t = 0 )
                #   nll: batch_size x beam_size (x 1)
                # nll becomes: batch_size x beam_size*vocab_size here
                # Reduce (N, K*V) to k-best
                nll, beam[tstep] = nll.unsqueeze_(2).add(log_p.view(
                    batch.size, -1, n_vocab)).view(batch.size, -1).topk(
                        k, sorted=False, largest=True)

                # previous indices into the beam and current token indices
                pdxs = beam[tstep] / n_vocab
                beam[tstep].remainder_(n_vocab)
                idxs = beam[tstep].view(-1)

                # Compute correct previous indices
                # Mask is needed since we're in flattened regime
                tile = pdxs.view(-1) + (nk_mask / k) * (k if tstep else 1)

                if tstep > 0:
                    # Permute all hypothesis history according to new order
                    beam[:tstep] = beam[:tstep].gather(2, pdxs.repeat(tstep, 1, 1))

            # Put an explicit <eos> to make idxs_to_sent happy
            beam[max_len - 1] = eos

            # Find lengths by summing tokens not in (pad,bos,eos)
            len_penalty = beam.gt(2).float().sum(0).clamp(min=1)

            if lp_alpha > 0.:
                len_penalty = ((5 + len_penalty)**lp_alpha) / 6**lp_alpha

            # Apply length normalization
            nll.div_(len_penalty)

            if n_best:
                # each elem is sample, then candidate
                tbeam = beam.permute(1, 2, 0).to('cpu').tolist()
                scores = nll.to('cpu').tolist()
                results.extend(
                    [(vocab.list_of_idxs_to_sents(b), s) for b, s in zip(tbeam, scores)])
            else:
                # Get best-1 hypotheses
                top_hyps = nll.topk(1, sorted=False, largest=True)[1].squeeze(1)
                hyps = beam[:, range(batch.size), top_hyps].t().to('cpu')
                results.extend(vocab.list_of_idxs_to_sents(hyps.tolist()))

        # Recover order of the samples if necessary
        return sort_predictions(data_loader, results)
