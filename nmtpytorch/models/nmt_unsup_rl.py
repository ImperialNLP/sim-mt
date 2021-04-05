# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn

from ..layers import TextEncoder
from ..layers.decoders import get_decoder
from ..optimizer import Optimizer
from ..utils.misc import get_n_params, load_pt_file
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..utils.data import sort_predictions
from ..datasets import MultimodalDataset
from ..metrics import Metric
from ..agents.hierarchical_agents.DIAYN import DIAYN
from ..utilities.data_structures.Config import Config
from ..environments.Trans_Env import Trans_Env

logger = logging.getLogger('nmtpytorch')


class DIAYN_Agent(nn.Module):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'emb_dim': 128,  # Source and target embedding sizes
            'emb_maxnorm': None,  # Normalize embeddings l2 norm to 1
            'emb_gradscale': False,  # Scale embedding gradients w.r.t. batch frequency
            'enc_dim': 256,  # Encoder hidden size
            'enc_proj_dim': None,  # Encoder final projection
            'enc_proj_activ': 'linear',  # Encoder final projection activation
            'enc_type': 'gru',  # Encoder type (gru|lstm)
            'enc_type': 'gru',  # Encoder type (gru|lstm)
            'enc_lnorm': False,  # Add layer-normalization to encoder output
            'enc_bidirectional': True,  # Whether the RNN encoder should be bidirectional
            'n_encoders': 1,  # Number of stacked encoders
            'dec_dim': 256,  # Decoder hidden size
            'dec_type': 'gru',  # Decoder type (gru|lstm)
            'dec_variant': 'scond',  # (cond|simplegru|vector)
            'att_type': 'mlp',  # Attention type (mlp|dot)
            'att_temp': 1.,  # Attention temperature
            'att_activ': 'tanh',  # Attention non-linearity (all torch nonlins)
            'att_mlp_bias': False,  # Enables bias in attention mechanism
            'att_bottleneck': 'ctx',  # Bottleneck dimensionality (ctx|hid)
            'att_transform_ctx': True,  # Transform annotations before attention
            'att_ctx2hid': True,  # Add one last FC layer on top of the ctx
            'dropout_emb': 0,  # Simple dropout to source embeddings
            'dropout_ctx': 0,  # Simple dropout to source encodings
            'dropout_out': 0,  # Simple dropout to decoder output
            'dropout_enc': 0,  # Intra-encoder dropout if n_encoders > 1
            'tied_emb': False,  # Share embeddings: (False|2way|3way)
            'direction': None,  # Network directionality, i.e. en->de
            'max_len': 80,  # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,  # A key like 'en' to define w.r.t which dataset
            # the batches will be sorted
            'bucket_order': None,  # Curriculum: ascending/descending/None
            'sampler_type': 'bucket',  # bucket or approximate
            'sched_sampling': 0,  # Scheduled sampling ratio
            'short_list': 0,  # Short list vocabularies (0: disabled)
            'out_logic': 'simple',  # 'simple' or 'deep' output
            'translator_type': 'gs',
            'translator_args': {},

        }

        self.actor_critic_agent_hyperparameters = {
            "Actor": {
                "learning_rate": 0.0004,
                "linear_hidden_units": [64, 64],
                "final_layer_activation": "Softmax",
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

            "Critic": {
                "learning_rate": 0.0003,
                "linear_hidden_units": [64, 64],
                "final_layer_activation": None,
                "batch_norm": False,
                "buffer_size": 1000,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

            "min_steps_before_learning": 1,
            "batch_size": 1,
            "discount_rate": 0.99,
            "mu": 0.0,  # for O-H noise
            "theta": 0.15,  # for O-H noise
            "sigma": 0.25,  # for O-H noise
            "action_noise_std": 0.2,  # for TD3
            "action_noise_clipping_range": 0.5,  # for TD3
            "update_every_n_steps": 1,
            "learning_updates_per_learning_session": 1,
            "automatically_tune_entropy_hyperparameter": False,
            "entropy_term_weight": 0.01,
            "add_extra_noise": False,
            "do_evaluation_iterations": True,
            "clip_rewards": False
        }

        self.dqn_agent_hyperparameters = {
            "learning_rate": 0.005,
            "batch_size": 128,
            "buffer_size": 10000,
            "epsilon": 1.0,
            "epsilon_decay_rate_denominator": 3,
            "discount_rate": 0.89,
            "tau": 0.01,
            "alpha_prioritised_replay": 0.6,
            "beta_prioritised_replay": 0.1,
            "incremental_td_error": 1e-8,
            "update_every_n_steps": 3,
            "linear_hidden_units": [30, 15],
            "final_layer_activation": "None",
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "clip_rewards": False
        }

        self.config_hyperparameters = {
            "Policy_Gradient_Agents": {
                "learning_rate": 0.05,
                "linear_hidden_units": [30, 15],
                "final_layer_activation": "TANH",
                "learning_iterations_per_round": 10,
                "discount_rate": 0.9,
                "batch_norm": False,
                "clip_epsilon": 0.2,
                "episodes_per_learning_round": 10,
                "normalise_rewards": True,
                "gradient_clipping_norm": 5,
                "mu": 0.0,
                "theta": 0.15,
                "sigma": 0.2,
                "epsilon_decay_rate_denominator": 1,
                "clip_rewards": False
            },

            "Actor_Critic_Agents": self.actor_critic_agent_hyperparameters,
            "DIAYN": {
                "DISCRIMINATOR": {
                    "learning_rate": 0.0001,
                    "linear_hidden_units": [30, 30],
                    "final_layer_activation": None,
                    "gradient_clipping_norm": 5

                },
                "AGENT": self.actor_critic_agent_hyperparameters,
                "num_skills": 4,
                "num_unsupervised_episodes": 1
            }
        }

    def __init__(self, opts):
        super().__init__()

        # opts -> config file sections {.model, .data, .vocabulary, .train}
        self.opts = opts

        # Vocabulary objects
        self.vocabs = {}

        # Each auxiliary loss should be stored inside this dictionary
        # in order to be taken into account by the mainloop for multi-tasking
        self.aux_loss = {}

        # Setup options
        self.opts.model = self.set_model_options(opts.model)

        # Parse topology & languages
        self.topology = Topology(self.opts.model['direction'])

        # Load vocabularies here
        print(self.opts.model['short_list'])
        for name, fname in self.opts.vocabulary.items():
            print(name)
            print(fname)
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

    def reset_parameters(self):
        for name, param in self.named_parameters():
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)
        # Reset padding embedding to 0
        for layer in ('enc', 'dec'):
            _layer = getattr(self, layer)
            if hasattr(_layer, 'emb'):
                with torch.no_grad():
                    _layer.emb.weight.data[0].fill_(0)

    def create_encoder(self):
        self.enc = TextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=self.n_src_vocab,
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

        self.ctx_sizes = {str(self.sl): self.enc.ctx_size}

    def create_decoder(self):
        Decoder = get_decoder('cond')

        self.dec = Decoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.sl),
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

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""
        self.create_encoder()
        self.create_decoder()

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.enc.emb.weight = self.dec.emb.weight

        config = Config()
        config.seed = 1
        config.environment = Trans_Env('trans_env')
        config.num_episodes_to_run = 10
        # config.file_to_save_data_results = "data_and_graphs/Hopper_Results_Data.pkl"
        # config.file_to_save_results_graph = "data_and_graphs/Hopper_Results_Graph.png"
        config.show_solution_score = False
        config.visualise_individual_results = False
        config.visualise_overall_agent_results = True
        config.standard_deviation_results = 1.0
        config.runs_per_agent = 3
        config.use_GPU = True
        config.hyperparameters = self.config_hyperparameters
        config.state_size = 840
        actor_optimizer = list(
            filter(lambda p: p[1].requires_grad, self.named_parameters()))
        params = [param for (name, param) in actor_optimizer]
        param_groups = [{'params': params}]
        self.agent = DIAYN(config, actor=self, actor_optimizer=param_groups)

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        self.dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            max_len=self.opts.model['max_len'],
            sampler_type=self.opts.model['sampler_type'],
            bucket_by=self.opts.model['bucket_by'],
            bucket_order=self.opts.model['bucket_order'],
            # order_file is for multimodal adv. evaluation
            order_file=self.opts.data[split + '_set'].get('ord', None))
        logger.info(self.dataset)
        return self.dataset

    def get_bos(self, batch_size):
        """Returns a representation for <bos> embeddings for decoding."""
        return torch.LongTensor(batch_size).fill_(self.trg_vocab['<bos>'])

    def cache_enc_states(self, batch):
        _ = self.enc(batch[self.sl])

    def get_enc_state_dict(self, batch=None, up_to=int(1e6)):
        return {str(self.sl): self.enc.get_states(up_to)}

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
        # Get loss dict
        self.agent.run_n_episodes(batch=batch)

        # result = self.dec(self.encode(batch), batch[self.tl])
        # result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]
        return {'loss': torch.zeros(1, requires_grad=True), 'n_items': 1}

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
                log_ps, h_ts, _, _ = zip(
                    *[f_next(cd, dec.get_emb(idxs), h_t[tile]) for
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
                len_penalty = ((5 + len_penalty) ** lp_alpha) / 6 ** lp_alpha

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
