from torch import nn


class Residual(nn.Module):

    def __init__(self, dropout=0.1):
        """
        Creates a Residual layer.
        :param dropout: The dropout rate.
        """
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, inputs):
        """
        Performs a forward pass over the Residual layer.
        :param inputs: The inputs, should be the residual x and f_x.
        :return: The output of the forward pass.
        """
        # Unpack into `x` and `Sublayer(x)`
        x, f_x = inputs
        return x + self.dropout_layer(f_x)
