import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm

from nmtpytorch.layers.transformers import TFEmbedding
from nmtpytorch.layers.transformers.decoder_block import DecoderBlock
from nmtpytorch.utils.mask_utils import generate_combined_mask, generate_lookahead_mask


class TFDecoder(nn.Module):
    """Decoder block for Transformer.

    Arguments:

    Input:

    Output:
    """

    def __init__(self, model_dim, ff_dim, n_heads, n_layers, num_embeddings, tied_emb=False, ff_activ='gelu',
                 dropout=0.1, attn_dropout=0.0, pre_norm=True, img_attn=None, n_mm_hier_heads=8,
                 store_attn_weights=True):
        """
        Creates a TFDecoder.
        :param model_dim: The model dimension.
        :param ff_dim: The feed-forward layer dimension.
        :param n_heads: The number of heads.
        :param n_layers: The number of layers.
        :param num_embeddings: The number of the embeddings.
        :param tied_emb: Whether to tie the input and output embeddings. Default: False.
        :param ff_activ: The feed forward layer activation function. Default 'gelu'.
        :param dropout: The dropout value. Default 0.1.
        :param pre_norm: Whether it should use 'pre_norm' layer types or 'post_norm' Default True.
        """
        super().__init__()
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_activ = ff_activ
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.store_attn_weights = store_attn_weights
        self.blocks = []

        self.trg_embedding = TFEmbedding(num_embeddings=num_embeddings, embedding_dim=self.model_dim, dropout=dropout)

        for _ in range(self.n_layers):
            layers = DecoderBlock(model_dim=self.model_dim, n_heads=self.n_heads, ff_dim=self.ff_dim,
                                  ff_activ=self.ff_activ, dropout=self.dropout, attn_dropout=attn_dropout,
                                  pre_norm=self.pre_norm, img_attn=img_attn, n_mm_hier_heads=n_mm_hier_heads)
            self.blocks.append(layers)

        self.blocks = nn.ModuleList(self.blocks)
        self._all_attention_weights = []

        self.final_layer_norm = None
        if self.pre_norm:
            self.final_layer_norm = LayerNorm(self.model_dim, eps=1e-6)

        self.output_layer = nn.Linear(self.model_dim, num_embeddings)

        if tied_emb:
            self.output_layer.weight = self.trg_embedding.weight

    def f_init(self, encoder_data):
        """
        Returns the initial hidden state of the decoder. N/A for the transformer.
        :param encoder_data:
        :return:
        """
        return None

    def forward(self, encoder_data, target, **kwargs):
        """Forward-pass of the decoder block.
        :param encoder_data: a tuple containing the encoder's hidden states tensor, shape (s_len, bsize, model_dim)
                             and the corresponding mask.
        :param target: input tensor, shape (t_len, bsize, model_dim)
        :param kwargs: Extra arguments for the decoder. In wait-k training, 'k' should be passed.

        :return: For backward compatibility with other decoders the method returns a tuple:
                the result from the final output layer and the decoders hidden states.
        """
        encoder_states, encoder_mask = encoder_data['src']
        encoder_image = self._get_image_data(encoder_data)

        encoder_mask = self._create_waitk_encoder_mask_if_needed(encoder_mask, encoder_states, kwargs, target)

        decoder_mask = generate_combined_mask(target)
        decoder_x = self.trg_embedding(target)

        self._all_attention_weights = []
        for block in self.blocks:
            decoder_x, attn_weights = block(encoder_states, decoder_x, encoder_mask, decoder_mask, encoder_image)
            if self.store_attn_weights is True:
                self._all_attention_weights.append(attn_weights)

        if self.pre_norm:
            decoder_x = self.final_layer_norm(decoder_x)

        return F.log_softmax(self.output_layer(decoder_x), dim=-1), decoder_x

    @staticmethod
    def _create_waitk_encoder_mask_if_needed(encoder_mask, encoder_states, kwargs, target):
        if 'k' in kwargs:
            simultaneous_k = kwargs['k']
            encoder_lookahead_mask = generate_lookahead_mask(encoder_states, simultaneous_k, target.shape[0])
            encoder_mask = encoder_mask | encoder_lookahead_mask
        return encoder_mask

    @staticmethod
    def _get_image_data(encoder_data):
        encoder_image = None
        if 'image' in encoder_data:
            encoder_image, _ = encoder_data['image']
        return encoder_image

    def f_next(self, encoder_data, next_word_emb, hidden_states, hypothesis):
        probs, decoder_x = self.forward(encoder_data, hypothesis)
        next_word_probs = probs[-1, :, :]

        return next_word_probs, decoder_x

    def get_emb(self, data):
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
        return self.trg_embedding(data)

    def get_attention_weights(self):
        return self._all_attention_weights
