from torch.nn import Module

from nmtpytorch.layers.transformers.sublayers.cross_attention_sublayer import CrossAttentionSublayer
from nmtpytorch.layers.transformers.sublayers.hierarchical_mm_cross_attention_sublayer import \
    HierarchicalMMCrossAttentionSublayer
from nmtpytorch.layers.transformers.sublayers.parallel_mm_cross_attention_sublayer import \
    ParallelMMCrossAttentionSublayer
from nmtpytorch.layers.transformers.sublayers.positionwise_sublayer import PositionWiseSublayer
from nmtpytorch.layers.transformers.sublayers.self_attention_sublayer import SelfAttentionSublayer
from nmtpytorch.layers.transformers.sublayers.serial_mm_cross_attention_sublayer import SerialMMCrossAttentionSublayer


class DecoderBlock(Module):

    def __init__(self, model_dim, n_heads, ff_dim, ff_activ='gelu', dropout=0.1, attn_dropout=0.0, pre_norm=True,
                 img_attn=None, n_mm_hier_heads=8):
        """
        Creates a decoder block, consisting of self attention, cross-attention and a position wise feed forward network.
        :param model_dim: The model dimensions.
        :param n_heads: The number of attention heads.
        :param ff_dim: The feed forward layer units.
        :param ff_activ: The feed forward layer activation function. Default 'gelu'.
        :param dropout: The dropout value. Default 0.1.
        :param img_attn: What kind of image attention should be applied; can be 'parallel', 'serial', or None.
                         Default None.
        """
        super().__init__()

        self.img_attn = img_attn
        self.self_attn = SelfAttentionSublayer(model_dim, n_heads, dropout, attn_dropout, pre_norm)
        self.cross_attn = self._create_cross_attn_layer(attn_dropout, dropout, img_attn, model_dim,
                                                        n_heads, pre_norm, n_mm_hier_heads)
        self.feed_forward = PositionWiseSublayer(model_dim, ff_dim, ff_activ, dropout, pre_norm)

    @staticmethod
    def _create_cross_attn_layer(attn_dropout, dropout, img_attn, model_dim, n_heads, pre_norm, n_hier_heads):
        if img_attn is not None and img_attn == 'parallel':
            return ParallelMMCrossAttentionSublayer(model_dim, n_heads, dropout, attn_dropout, pre_norm)
        elif img_attn is not None and img_attn == 'serial':
            return SerialMMCrossAttentionSublayer(model_dim, n_heads, dropout, attn_dropout, pre_norm)
        elif img_attn is not None and img_attn == 'hierarchical':
            return HierarchicalMMCrossAttentionSublayer(model_dim, n_heads, dropout, attn_dropout,
                                                        pre_norm, n_hier_heads)
        else:
            return CrossAttentionSublayer(model_dim, n_heads, dropout, attn_dropout, pre_norm)

    def forward(self, encoder_x, decoder_x, encoder_mask=None, decoder_mask=None, image_x=None):
        all_weights = {}
        decoder_x, all_weights['self'] = self.self_attn(decoder_x, decoder_mask)

        decoder_x_attn, all_weights['cross'] = self.cross_attn(decoder_x, encoder_x, encoder_x, encoder_mask,
                                                               key_img=image_x, value_img=image_x)

        return self.feed_forward(decoder_x_attn, decoder_mask), all_weights
