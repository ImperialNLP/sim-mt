import logging

import torch
from torch import nn
from ..layers.transformers import MultiheadAttention

from . import SimultaneousNMT


logger = logging.getLogger('nmtpytorch')


class EncoderSelfAttentionSimultaneousNMT(SimultaneousNMT):
    """Simultaneous self-attentive MMT."""

    def set_defaults(self):
        super().set_defaults()
        self.defaults.update({
            'n_heads': 1,
            'att_dropout': 0.0,
        })

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""
        encoders = {}
        for key in self.topology.srcs.keys():
            encoders[key] = getattr(self, f'create_{key}_encoder')()

        # Separate out visual encoder to avoid multimodal decoder-side
        # attention to be enabled
        self.ff_vis_enc = encoders.pop('image')

        self.encoders = nn.ModuleDict(encoders)
        self.dec = self.create_decoder(encoders=self.encoders)

        if self.opts.model['feat_mode']:
            if self.opts.model['feat_mode'].endswith('objs'):
                self.obj_emb = nn.Embedding(1600, self.opts.model['emb_dim'])

            if self.opts.model['feat_mode'].startswith('attrs'):
                self.attr_emb = nn.Embedding(400, self.opts.model['emb_dim'])

        # create the cross-modal self-attention network
        self.mm_attn = MultiheadAttention(
            self.opts.model['enc_dim'], self.opts.model['enc_dim'],
            n_heads=self.opts.model['n_heads'],
            dropout=self.opts.model['att_dropout'], attn_type='cross')
        self.mm_lnorm = nn.LayerNorm(self.opts.model['enc_dim'])

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.encoders[str(self.sl)].emb.weight = self.dec.emb.weight

    def cache_enc_states(self, batch):
        """Caches encoder states internally by forward-pass'ing each encoder."""
        if self.opts.model['feat_mode']:
            # Prepare category embeddings before if required
            if self.opts.model['feat_mode'].endswith('objs'):
                obj_idxs, *attr_idxs = torch.split(batch['image'], 1, dim=-1)
                embs = self.obj_emb(obj_idxs.squeeze(-1))
                if len(attr_idxs) == 1:
                    attr_embs = self.attr_emb(attr_idxs[0].squeeze(-1))
                    embs = torch.cat((embs, attr_embs), -1)

                # replace image features in the batch
                batch['image'] = embs

        self.encoders['src'](batch['src'])
        self.ff_vis_enc(batch['image'])

        src_states, src_mask = self.encoders['src'].get_states()
        img_states, img_mask = self.ff_vis_enc.get_states()

        # key values are image states
        kv = img_states.transpose(0, 1)
        attn_out = self.mm_attn(
            q=src_states.transpose(0, 1), k=kv, v=kv,
            q_mask=src_mask.transpose(0, 1).logical_not()).transpose(0, 1)

        # Inject this into the encoder itself for caching
        self.encoders['src']._states = self.mm_lnorm(src_states + attn_out)

    def get_enc_state_dict(self, up_to=int(1e6)):
        """Encodes the batch optionally by partial encoding up to `up_to`
        words for derived simultaneous NMT classes. By default, the value
        is large enough to leave it as vanilla NMT."""
        return {str(k): e.get_states(up_to=up_to) for k, e in self.encoders.items()}
