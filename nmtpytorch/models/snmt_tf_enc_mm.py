from nmtpytorch.layers.transformers import TFEncoder
from nmtpytorch.models import SimultaneousTFNMT


class EncoderMMSimultaneousTFNMT(SimultaneousTFNMT):

    def __init__(self, opts):
        super().__init__(opts)
        assert not self.opts.model['enc_bidirectional'], \
            'Bidirectional TF encoder is not currently supported for simultaneous MT.'

    def set_defaults(self):
        super().set_defaults()
        self.defaults.update({
            # Decoding/training simultaneous NMT args
            'enc_img_attn': 'flat',  # The image attention applied at the encoder. Default: 'flat'.
        })

    def _create_src_encoder(self):
        """
        Returns a transformer encoder.
        :return: a transformer encoder.
        """
        return TFEncoder(
            model_dim=self.opts.model['model_dim'],
            n_heads=self.opts.model['num_heads'],
            ff_dim=self.opts.model['enc_ff_dim'],
            n_layers=self.opts.model['enc_n_layers'],
            num_embeddings=self.n_src_vocab,
            ff_activ=self.opts.model['ff_activ'],
            dropout=self.opts.model['dropout'],
            attn_dropout=self.opts.model['attn_dropout'],
            pre_norm=self.opts.model['pre_norm'],
            enc_bidirectional=self.opts.model['enc_bidirectional'],
            enc_img_attn=self.opts.model['enc_img_attn']
        )

    def cache_enc_states(self, batch, **kwargs):
        """
        Caches the encoder states. It first obtains the visual information from the image encoder and passes it to the
        textual encoder.
        :param batch: The batch.
        :param kwargs: Any additional args.
        """
        visual_x = self.encoders['image'](batch['image'])
        _ = self.encoders['src'](batch['src'], img_data=visual_x)

    def get_enc_state_dict(self, up_to=int(1e6)):
        """
        Returns the cached encoder states. As the visual ones are included in the textual ones, only the textual encoder
        states are used.
        :param up_to:  The amount of timesteps to return.
        :return: The encoder states up to a certain timestep.
        """
        return {'src': self.encoders['src'].get_states(up_to=up_to)}
