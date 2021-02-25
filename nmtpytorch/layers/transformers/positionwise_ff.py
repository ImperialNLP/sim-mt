from torch import nn

from .. import FF


class PositionwiseFF(nn.Module):
    """Positionwise Feed-forward layer.

    Arguments:

    Input:

    Output:
    """

    def __init__(self, model_dim, ff_dim, activ='gelu', dropout=0.1):
        """
        Creates a PositionwiseFF.
        :param model_dim: The model dimensions.
        :param ff_dim: The feedforward dimensions.
        :param activ: The activation function. Default: gelu
        :param dropout: The amount of dropout. Default: 0.1
        """
        super().__init__()
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.activ = activ

        # Create the layers
        self.layers = nn.Sequential(
            FF(self.model_dim, self.ff_dim, activ=self.activ),
            nn.Dropout(dropout),
            FF(self.ff_dim, self.model_dim, activ=None),
        )

    def forward(self, x):
        return self.layers(x)
