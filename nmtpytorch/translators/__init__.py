from .greedy import GreedySearch
from .sim_greedy import SimultaneousGreedySearch
from .waitk_greedy import SimultaneousWaitKGreedySearch
from .simrl_greedy import SimultaneousRLGreedySearch


def get_translator(_type):
    return {
        'gs': GreedySearch,
        'sgs': SimultaneousGreedySearch,
        'wk': SimultaneousWaitKGreedySearch,
        'srlgs': SimultaneousRLGreedySearch,
    }[_type]
