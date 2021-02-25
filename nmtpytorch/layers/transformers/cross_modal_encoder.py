from torch import sigmoid
from torch.nn import Linear, LayerNorm, Dropout

from nmtpytorch.layers import ScaledDotAttention
from nmtpytorch.layers.transformers.visual_object_tags_encoder import VisualObjectTagsEncoder


class CrossModalEncoder(VisualObjectTagsEncoder):

    def __init__(self, input_size, proj_dim=None, proj_activ=None, layer_norm=False, l2_norm=False,
                 dropout=0.0, feat_mode=None, attr_embeddings_vocab_size=400,
                 attrs_embeddings_dim=512, objs_embeddings_vocab_size=1600, objs_embeddings_dim=512,
                 model_dim=512, mm_attn_heads=8, attn_dropout=0.0, fusion='sum', fusion_lnorm=True, fusion_dropout=0.0,
                 image_masking=False, alignment_head=None, boxes_dim=None):
        """
        Creates a CrossModalEncoder.
        :param input_size: The input size.
        :param proj_dim: The projection dimensionality for the visual features.
        :param proj_activ: The activation function to be applied on the projection.
        :param layer_norm: Whether to apply layer normalisation. Default: False.
        :param l2_norm: Whether to apply l2 normalisation. Default False.
        :param dropout: The dropout value for the visual projection.
        :param feat_mode: The feature mode. Can be None, 'attrs', 'objs' or 'attrs+objs'. Default: None.
        :param attr_embeddings_vocab_size: The size of of the attribute embeddings vocabulary. Default: 400.
                                           If feat_mode is None, it is ignored.
        :param attrs_embeddings_dim: The attribute embedding dimensions. Default: 512.
                                     If feat_mode is None, it is ignored.
        :param objs_embeddings_vocab_size: The size of of the objects embeddings vocabulary. Default: 1600.
                                           If feat_mode is None, it is ignored.
        :param objs_embeddings_dim: The object embedding dimensions. Default: 512.
                                    If feat_mode is None, it is ignored.
        :param model_dim: The model dimensions. Default: 512.
        :param mm_attn_heads: The number of multimodal attention heads. Default: 8.
        :param attn_dropout: The attention dropout. Default 0.0.
        :param fusion: The fusion. Can be 'gate' or 'sum'. Default 'sum'.
        :param fusion_lnorm: Whether to apply layer normalisation after fusing both modalities. Default: True.
        :param fusion_dropout: The amount of dropout after fusing the two modalities. Default: 0.0.
        """
        super().__init__(input_size=input_size, proj_dim=proj_dim, proj_activ=proj_activ, layer_norm=layer_norm,
                         l2_norm=l2_norm, dropout=dropout, feat_mode=feat_mode,
                         attr_embeddings_vocab_size=attr_embeddings_vocab_size,
                         attrs_embeddings_dim=attrs_embeddings_dim,
                         objs_embeddings_vocab_size=objs_embeddings_vocab_size,
                         objs_embeddings_dim=objs_embeddings_dim, image_masking=image_masking, boxes_dim=boxes_dim)

        self.cross_attn = ScaledDotAttention(model_dim, mm_attn_heads, attn_dropout)

        self.fusion = fusion
        if fusion == 'gate':
            self.gate_visual = Linear(proj_dim, 1, bias=False)
            self.gate_txt = Linear(model_dim, 1, bias=False)
        self.fusion_dropout = Dropout(fusion_dropout)
        if fusion_lnorm is True:
            self.fusion_lnorm = LayerNorm(self.ctx_size)
        self._enc_txt_mask = None
        self._all_attention_weights = []
        self._alignment_weights = None
        self._alignment_head = alignment_head

    def forward(self, x, **kwargs):
        """
        Performs a forward pass.
        :param x: The input.
        :param kwargs: The additional arguments. Should include the 'enc_txt' for the CrossModalEncoder encoder.
        :return: The output of the forward pass.
        """
        self._all_attention_weights = []
        image_x, image_mask = super().forward(x)

        enc_txt_x, self._enc_txt_mask = kwargs['enc_txt']
        enc_img_x, img_attn_weights = self.cross_attn((enc_txt_x, image_x, image_x, image_mask))

        self._all_attention_weights.append({'cross': img_attn_weights})
        if self._alignment_head is not None:
            self._alignment_weights = img_attn_weights[:, self._alignment_head, :, :]

        self._states = self._fuse_context(enc_txt_x, enc_img_x)

        return self._states, self._mask

    def _fuse_context(self, enc_txt, enc_img):
        # Gating mechanism following the approach described in this paper: https://openreview.net/pdf?id=Byl8hhNYPS
        if self.fusion == 'gate':
            gate_lambda = sigmoid(self.gate_visual(enc_img) + self.gate_txt(enc_txt))
            combined = enc_txt + self.fusion_dropout((gate_lambda * enc_img))
        else:
            combined = enc_txt + self.fusion_dropout(enc_img)
        if self.fusion_lnorm is not None:
            combined = self.fusion_lnorm(combined)
        return combined

    def get_states(self, up_to=int(1e6)):
        """
        Retrieves the states. For the cross modal encoder it will return the states up to the value supplied.
        :param up_to: The amount of tokens to return.
        :return: The states up to the specified value.
        """
        assert self._states is not None, \
            "encoder was not called for caching the states."
        return self._states[:up_to], self._enc_txt_mask[:, :, :up_to]

    def get_attention_weights(self):
        return self._all_attention_weights

    def get_alignment_weights(self):
        return self._alignment_weights
