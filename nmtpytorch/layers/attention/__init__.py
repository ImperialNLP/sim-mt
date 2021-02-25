from .mlp import MLPAttention
from .dot import DotAttention
from .hierarchical import HierarchicalAttention
from .uniform import UniformAttention
from .scaled_dot import ScaledDotAttention


def get_attention(type_):
    return {
        'mlp': MLPAttention,
        'dot': DotAttention,
        'hier': HierarchicalAttention,
        'uniform': UniformAttention,
        'scaled_dot': ScaledDotAttention,
    }[type_]
