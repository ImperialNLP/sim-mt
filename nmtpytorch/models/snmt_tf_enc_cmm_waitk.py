from nmtpytorch.layers.transformers.cross_modal_encoder import CrossModalEncoder
from nmtpytorch.models import SimultaneousTFWaitKNMT


class EncoderCrossMMSimultaneousTFWaitKNMT(SimultaneousTFWaitKNMT):

    def __init__(self, opts):
        super().__init__(opts)
        assert not self.opts.model['enc_bidirectional'], \
            'Bidirectional TF encoder is not currently supported for simultaneous MT.'

    def set_defaults(self):
        super().set_defaults()
        self.defaults.update({
            # Decoding/training simultaneous NMT args
            'enc_fusion': 'sum',        # The encoder fusion type.Can be: 'sum' or 'gate'. Default 'sum'.
            'enc_fusion_lnorm': True,   # Whether to apply layer normalization after fusing the encoder.
            'mm_attn_heads': 8,         # The number of multimodal attention heads.
            'enc_fusion_dropout': 0.0,  # The amount of dropout after the fusion.
        })

    def _create_image_encoder(self):
        return CrossModalEncoder(
            input_size=self.opts.model['aux_dim'],
            proj_dim=self.opts.model['aux_proj_dim'],
            proj_activ=self.opts.model['aux_proj_activ'],
            layer_norm=self.opts.model['aux_lnorm'],
            l2_norm=self.opts.model['aux_l2norm'],
            dropout=self.opts.model['aux_dropout'],
            feat_mode=self.opts.model['feat_mode'],
            model_dim=self.opts.model['model_dim'],
            mm_attn_heads=self.opts.model['mm_attn_heads'],
            attn_dropout=self.opts.model['attn_dropout'],
            fusion=self.opts.model['enc_fusion'],
            fusion_lnorm=self.opts.model['enc_fusion_lnorm'],
            fusion_dropout=self.opts.model['enc_fusion_dropout'],
            boxes_dim=self.opts.model['img_boxes_dim']
        )

    def cache_enc_states(self, batch, **kwargs):
        """
        Caches the encoder hidden states, by first computing the textual hidden states, and then combining them with the
        visual encoder using the cross modal encoder.
        :param batch: The batch.
        :param kwargs: Any additional args.
        """
        enc_txt = self.encoders['src'](batch['src'])
        _ = self.encoders['image'](batch['image'], enc_txt=enc_txt)

    def get_enc_state_dict(self, up_to=int(1e6)):
        """
        Get the encoder states. In the cross modal case retrive the ones from the cross modal image encoder, as they
        also contain the textual encoder hidden states.
        :param up_to: The amount of timesteps to return.
        :return: The encoder states up to a certain timestep.
        """
        return {'src': self.encoders['image'].get_states(up_to=up_to)}
