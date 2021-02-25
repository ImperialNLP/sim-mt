from .conditional import ConditionalDecoder
from .sconditional import SimultaneousConditionalDecoder
from .conditionalmm import ConditionalMMDecoder
from .conditionalrl import ConditionalDecoderRL


def get_decoder(type_):
    """Only expose ones with compatible __init__() arguments for now."""
    return {
        'cond': ConditionalDecoder,
        'scond': SimultaneousConditionalDecoder,
        'condmm': ConditionalMMDecoder,
        'condrl': ConditionalDecoderRL,
    }[type_]
