from nmtpytorch.layers.transformers import PositionwiseFF
from nmtpytorch.layers.transformers.sublayers.base_sublayer import BaseSublayer


class PositionWiseSublayer(BaseSublayer):

    def __init__(self, model_dim, ff_dim, ff_activ='gelu', dropout=0.1, is_pre_norm=False):
        """
        Creates a PositionWiseSublayer.
        :param model_dim: The model dimensions.
        :param ff_dim: The dimensions of the feed forward network.
        :param ff_activ: The activation of the feed forward network.
        :param dropout: The dropout rate.
        :param is_pre_norm: Whether the layer type is pre_norm. Default: True.
        """
        super().__init__(model_dim, dropout, is_pre_norm)
        self.feed_forward = PositionwiseFF(model_dim, ff_dim, ff_activ, dropout=dropout)

    def forward(self, x, mask=None):
        """
        Performs a forward pass over the PositionWiseSublayer.
        :param x: The input x.
        :param mask: The input mask.
        :return: The output from the forward pass of the PositionWiseSublayer.
        """
        residual = x
        x = self.apply_pre_norm_if_needed(x)
        x = self.feed_forward(x)
        x = self.apply_residual(residual, x)
        x = self.apply_post_norm_if_needed(x)
        return x
