from .metric import Metric
from .multibleu import BLEUScorer
from .sacrebleu import SACREBLEUScorer
from .meteor import METEORScorer
from .wer import WERScorer
from .cer import CERScorer
from .rouge import ROUGEScorer
from .simnmt import AVPScorer, AVLScorer, Q2AVPScorer, Q2AVLScorer

beam_metrics = ["BLEU", "SACREBLEU", "METEOR", "WER", "CER", "ROUGE", "Q2AVL", "Q2AVP"]

metric_info = {
    'BLEU': 'max',
    'SACREBLEU': 'max',
    'METEOR': 'max',
    'ROUGE': 'max',
    'LOSS': 'min',
    'WER': 'min',
    'CER': 'min',
    'ACC': 'max',
    'RECALL': 'max',
    'PRECISION': 'max',
    'F1': 'max',
    # simultaneous translation
    'AVP': 'min',   # Average proportion (Cho and Esipova, 2016)
    'AVL': 'min',   # Average Lagging (Ma et al., 2019 (STACL))
    'Q2AVP': 'max',
    'Q2AVL': 'max'
}
