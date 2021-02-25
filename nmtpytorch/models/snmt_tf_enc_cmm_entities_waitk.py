from nmtpytorch.layers.transformers.cross_modal_encoder import CrossModalEncoder
from nmtpytorch.models import SimultaneousTFWaitKNMT
from nmtpytorch.utils.supervised_attention_loss import SupervisedAttentionLoss


class EncoderCrossMMEntitiesSimultaneousTFWaitKNMT(SimultaneousTFWaitKNMT):

    def __init__(self, opts):
        super().__init__(opts)
        assert not self.opts.model['enc_bidirectional'], \
            'Bidirectional TF encoder is not currently supported for simultaneous MT.'
        self.aux_loss = {}
        self._aux_loss_function = None

    def set_defaults(self):
        super().set_defaults()
        self.defaults.update({
            # Decoding/training simultaneous NMT args
            'translator_type': 'wk',  # This model implements train-time wait-k
            'translator_args': {'k': 1e4},  # k as in wait-k in training
            'consecutive_warmup': 0,  # consecutive training for this many epochs
            'enc_fusion': 'sum',            # The encoder fusion type.Can be: 'sum' or 'gate'. Default 'sum'.
            'enc_fusion_lnorm': True,       # Whether to apply layer normalization after fusing the encoder.
            'mm_attn_heads': 8,             # The number of multimodal attention heads.
            'enc_fusion_dropout': 0.0,      # The amount of dropout after the fusion.
            'image_masking': True,          # Whether to mask invalid image regions, which have only zero feature vectors.
            'alignment_head': 0,            # The alignment head to supervise. Default 0.
            'supervised_loss_func': 'nll',  # The supervised loss function. Default 'nll'.
            'aux_loss_weight': 1.0          # The auxiliary loss weight. Default 1.

        })

    def setup(self, is_train=True):
        super(EncoderCrossMMEntitiesSimultaneousTFWaitKNMT, self).setup(is_train=is_train)
        self._aux_loss_function = SupervisedAttentionLoss(self.opts.model['supervised_loss_func'],
                                                          loss_weight=self.opts.model['aux_loss_weight'])

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
            image_masking=self.opts.model['image_masking'],
            alignment_head=self.opts.model['alignment_head'],
            boxes_dim=self.opts.model['img_boxes_dim']
        )

    def get_attention_weights(self):
        return {'encoder_src': self.encoders['src'].get_attention_weights(),
                'encoder_img': self.encoders['image'].get_attention_weights(),
                'decoder': self.dec.get_attention_weights()}

    def _create_alignments_encoder(self):
        pass

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
        Get the encoder states. In the cross modal case retrieve the ones from the cross modal image encoder, as they
        also contain the textual encoder hidden states.
        :param up_to: The amount of timesteps to return.
        :return: The encoder states up to a certain timestep.
        """
        return {'src': self.encoders['image'].get_states(up_to=up_to)}

    def get_alignment_weights(self):
        return self.encoders['image'].get_alignment_weights()

    def forward(self, batch, **kwargs):
        """
        Performs a forward pass.
        :param batch: The batch.
        :param kwargs: Any extra arguments.
        :return: The output from the forward pass.
        """
        loss = super().forward(batch, **kwargs)

        self.aux_loss['attn_loss'] = self._aux_loss_function(self.get_alignment_weights(),
                                                             batch['alignments'].permute(1, 0, 2))

        return loss
