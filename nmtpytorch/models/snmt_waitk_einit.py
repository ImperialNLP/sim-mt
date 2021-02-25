import logging

from torch import nn

from ..layers.decoders import SimultaneousConditionalDecoder

from . import SimultaneousWaitKNMT

logger = logging.getLogger('nmtpytorch')

"""This is a modified version of the WaitK model in which visual interaction
is explored on encoder side, rather than in the decoder. The vectorial
visual features should be explicitly named `image` in the config file."""


class SimultaneousWaitKEncInitNMT(SimultaneousWaitKNMT):

    def set_defaults(self):
        super().set_defaults()

        # disable decoder-side multimodality
        for arg in ('mm_fusion_op', 'mm_fusion_dropout'):
            if arg in self.opts.model:
                del self.opts.model[arg]

    def create_decoder(self, encoders):
        """Creates and returns the RNN decoder. No hidden state initialization
        for sake of simplicity."""
        return SimultaneousConditionalDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            encoders=encoders,
            rnn_type=self.opts.model['dec_type'],
            tied_emb=self.opts.model['tied_emb'],
            att_type=self.opts.model['att_type'],
            att_temp=self.opts.model['att_temp'],
            att_activ=self.opts.model['att_activ'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            out_logic=self.opts.model['out_logic'],
            dec_inp_activ=self.opts.model['dec_inp_activ'],
        )

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""
        encoders = {}
        for key in self.topology.srcs.keys():
            encoders[key] = getattr(self, f'create_{key}_encoder')()

        ###############################
        # Separate this out for EncInit
        ###############################
        self.ff_vis_enc = encoders.pop('image')

        self.encoders = nn.ModuleDict(encoders)
        self.dec = self.create_decoder(encoders=self.encoders)

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.encoders[str(self.sl)].emb.weight = self.dec.emb.weight

    def cache_enc_states(self, batch):
        """Caches encoder states internally by forward-pass'ing each encoder."""
        # Project visual features (no mask involved)
        vis_proj, _ = self.ff_vis_enc(batch['image'])

        # Initialize the encoder with visual features
        self.encoders[self.sl](batch[self.sl], hx=vis_proj)
