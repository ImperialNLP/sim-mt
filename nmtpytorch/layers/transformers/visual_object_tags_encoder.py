from torch import nn, split, stack

from nmtpytorch.utils.mask_utils import generate_visual_features_padding_masks
from .. import FF
from .. import LayerNorm


class VisualObjectTagsEncoder(nn.Module):

    def __init__(self, input_size, proj_dim=None, proj_activ=None, layer_norm=False, l2_norm=False,
                 dropout=0.0, feat_mode=None, attr_embeddings_vocab_size=400, attrs_embeddings_dim=512,
                 objs_embeddings_vocab_size=1600, objs_embeddings_dim=512, image_masking=False, boxes_dim=None):
        """
        Creates a VisualObjectTagsEncoder.
        :param input_size: The input size.
        :param proj_dim: The projection dimensionality for the visual features.
        :param proj_activ: The activation function to be applied on the projection.
        :param layer_norm: Whether to apply layer normalisation. Default: False.
        :param l2_norm: Whether to apply l2 normalisation. Default False.
        :param dropout: The dropout value.
        :param feat_mode: The feature mode. Can be 'attrs', 'objs' or 'attrs+objs'.
        :param attr_embeddings_vocab_size: The size of of the attribute embeddings vocabulary. Default: 400.
        :param attrs_embeddings_dim: The attribute embedding dimensions. Default: 512.
        :param objs_embeddings_vocab_size: The size of of the objects embeddings vocabulary. Default: 1600.
        :param objs_embeddings_dim: The object embedding dimensions. Default: 512.
        """
        super().__init__()

        self.input_size = input_size
        self.boxes_dim = boxes_dim
        self.ctx_size = input_size
        self.l2_norm = l2_norm
        self._image_masking = image_masking

        self.feat_mode = feat_mode
        self.attr_embeddings = None
        self.obj_embeddings = None

        if self.feat_mode is not None:
            if 'attrs' in feat_mode:
                self.attr_embeddings = nn.Embedding(attr_embeddings_vocab_size, attrs_embeddings_dim)
            if 'objs' in feat_mode:
                self.obj_embeddings = nn.Embedding(objs_embeddings_vocab_size, objs_embeddings_dim)

        self.proj_layer = None
        if proj_dim is not None:
            self.proj_layer = FF(input_size, proj_dim, activ=proj_activ)
            self.ctx_size = proj_dim

        self.geometric_emb_layer = None
        if boxes_dim is not None:
            self.geometric_emb_layer = nn.Linear(boxes_dim, proj_dim)

        self.layer_norm_layer = None
        if layer_norm:
            self.layer_norm_layer = LayerNorm(self.ctx_size)

        self.dropout_layer = None
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)

        # Variables for caching
        self._states, self._mask = None, None

    def forward(self, x, **kwargs):
        """
        Performs a forward pass.
        :param x: The input.
        :param kwargs:
        :return: The output of the forward pass.
        """
        if self._image_masking:
            self._mask = generate_visual_features_padding_masks(x)

        boxes = None
        if self.feat_mode is not None and self.feat_mode in ['attrs', 'objs']:
            x = self.compute_obj_attr_embeddings(x)

        elif self.feat_mode is not None and self.feat_mode == 'roi_feats+boxes':
            x, boxes = split(x, [self.input_size, self.boxes_dim], dim=-1)

        if self.l2_norm:
            x.div_(x.norm(p=2, dim=-1, keepdim=True))

        if self.proj_layer:
            x = self.proj_layer(x)

        if self.geometric_emb_layer and boxes is not None:
            geometric_emb = self.geometric_emb_layer(boxes)
            x = x + geometric_emb

        if self.layer_norm_layer:
            x = self.layer_norm_layer(x)

        if self.dropout_layer:
            x = self.dropout_layer(x)

        self._states = x
        return self._states, self._mask

    def compute_obj_attr_embeddings(self, x):
        obj_idxs, *attr_idxs = split(x, 1, dim=-1)
        obj_embeddings = None
        if self.obj_embeddings is not None:
            obj_embeddings = self.obj_embeddings(obj_idxs.squeeze(-1))
        if self.attr_embeddings is not None and len(attr_idxs) == 1:
            attr_embeddings = self.attr_embeddings(attr_idxs[0].squeeze(-1))
            if obj_embeddings is not None:
                # Combine the embeddings so that we have: attr, object for every pair
                # Output shape: (2 * num_objects, batch_size, embedding_dim)
                x = stack((attr_embeddings, obj_embeddings), 1).reshape(-1, obj_embeddings.shape[1],
                                                                        obj_embeddings.shape[2])
            else:
                x = attr_embeddings
        return x

    def get_states(self, up_to=int(1e6)):
        assert self._states is not None, \
            "encoder was not called for caching the states."
        return self._states, self._mask


