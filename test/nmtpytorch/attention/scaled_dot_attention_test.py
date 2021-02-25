from torch.nn import MultiheadAttention
import torch
import numpy as np


from nmtpytorch.layers import ScaledDotAttention


def test_scale_dot_attention_same_size_encoder_decoder():
    model_dim = 16
    decoder_x = torch.rand(2, 4, model_dim)
    encoder_states = torch.rand(2, 4, model_dim)

    validate_attention(decoder_x, encoder_states, model_dim)


def test_scale_dot_attention_diff_size_encoder_decoder():
    model_dim = 16
    decoder_x = torch.rand(2, 4, model_dim)
    encoder_states = torch.rand(1, 4, model_dim)

    validate_attention(decoder_x, encoder_states, model_dim)


def test_scale_dot_attention_padding_mask_decoder():
    model_dim = 16
    decoder_x = torch.rand(3, 4, model_dim)
    encoder_states = torch.rand(2, 4, model_dim)
    encoder_states_mask = torch.randint(0, 2, (1, 3, 2)).bool()

    validate_attention(decoder_x, encoder_states, model_dim, encoder_states_mask)


def test_scale_dot_attention_lookahead_mask_decoder():
    model_dim = 16
    decoder_x = torch.rand(3, 4, model_dim)
    decoder_lookahead_mask = np.triu(np.ones((1, 3, 3)), k=1)
    decoder_lookahead_mask = torch.from_numpy(decoder_lookahead_mask) == 0

    validate_attention(decoder_x, decoder_x, model_dim, decoder_lookahead_mask)


def validate_attention(decoder_x, encoder_states, model_dim, mask=None):
    library_attn = MultiheadAttention(embed_dim=model_dim, num_heads=8)
    lib_mask = None
    # The mask in this Pytorch version is simply added to the vector, so we need to multiply it by
    # a large negative value, to push the softmax values towards zero.
    if mask is not None:
        lib_mask = mask.squeeze(0) * -1e8
    expected, _ = library_attn(decoder_x, encoder_states, encoder_states, attn_mask=lib_mask)
    scaled_dot = ScaledDotAttention(model_dim=model_dim, n_heads=8)
    set_same_weights(library_attn, scaled_dot, model_dim)
    out, _ = scaled_dot((decoder_x, encoder_states, encoder_states, mask))

    result = (torch.isclose(expected, out) == False).sum() == 0
    print("Expected and actual values are the same", result)
    assert result


def set_same_weights(source_layer, target_layer, model_dim):
    target_layer.lin_q.weight.data = source_layer.in_proj_weight[0:model_dim, :]
    target_layer.lin_k.weight.data = source_layer.in_proj_weight[model_dim:2*model_dim, :]
    target_layer.lin_v.weight.data = source_layer.in_proj_weight[2*model_dim:3*model_dim, :]
    target_layer.lin_o.weight.data = source_layer.out_proj.weight.data


test_scale_dot_attention_same_size_encoder_decoder()
test_scale_dot_attention_diff_size_encoder_decoder()
test_scale_dot_attention_padding_mask_decoder()
test_scale_dot_attention_lookahead_mask_decoder()
