from torch.nn import Module

from nmtpytorch.layers.transformers.sublayers.flat_mm_cross_attention_sublayer import FlatMMCrossAttentionSublayer
from nmtpytorch.layers.transformers.sublayers.positionwise_sublayer import PositionWiseSublayer
from nmtpytorch.layers.transformers.sublayers.self_attention_sublayer import SelfAttentionSublayer


class EncoderBlock(Module):

    def __init__(self, model_dim, ff_dim, n_heads, ff_activ='gelu', dropout=0.1, attn_dropout=0.0, pre_norm=True,
                 enc_img_attn=None):
        """
        Creates an EncoderBlock, consisting of a self-attention sublayer and a position-wise feed forward sublayer.
        :param model_dim: The model dimensions.
        :param ff_dim: The feed forward network dimensions.
        :param n_heads: The number of attention heads.
        :param ff_activ: The feed forward network activation function.
        :param dropout: The dropout.
        :param pre_norm: Whether it should use 'pre_norm' layer types or 'post_norm' Default True.
        :param enc_img_attn: The encoder image attention. Possible values: ['flat', 'None']. Default: None.
        """
        super().__init__()
        self.enc_img_attn = enc_img_attn
        if enc_img_attn == 'flat':
            self.multimodal_attn = FlatMMCrossAttentionSublayer(model_dim, n_heads, dropout, attn_dropout, pre_norm)
        else:
            self.self_attn = SelfAttentionSublayer(model_dim, n_heads, dropout, attn_dropout, pre_norm)
        self.feed_forward = PositionWiseSublayer(model_dim, ff_dim, ff_activ, dropout, pre_norm)

    def forward(self, encoder_x, encoder_mask=None, image_x=None, image_mask=None):
        """
        Performs a forward pass of an encoder block.
        :param encoder_x: The encoder's source text input.
        :param encoder_mask: The encoder's source text input mask.
        :param image_x: The encoder's image input.
        :param image_mask: The encoder's image input mask.

        :return: The output of the forward pass.
        """
        if self.enc_img_attn == 'flat' and image_x is not None:
            encoder_x, attn_weights = self.multimodal_attn(encoder_x,
                                                           key_txt=None, value_txt=None, mask_txt=encoder_mask,
                                                           key_img=image_x, value_img=image_x, mask_img=image_mask)

            all_attn_weights = {'multimodal': attn_weights}
        else:
            encoder_x, attn_weights = self.self_attn(encoder_x, encoder_mask)
            all_attn_weights = {'self': attn_weights}
        return self.feed_forward(encoder_x, encoder_mask), all_attn_weights
