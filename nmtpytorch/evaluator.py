# -*- coding: utf-8 -*-
from collections import OrderedDict

from . import metrics
from .utils.filterchain import FilterChain
from .utils.misc import get_language


class Evaluator:
    def __init__(self, refs, beam_metrics, filters=''):
        # metrics: list of upper-case beam-search metrics
        self.kwargs = {}
        self.scorers = OrderedDict()
        self.simmt_scorers = OrderedDict()
        self.refs = list(refs.parent.glob(refs.name))
        self.language = get_language(self.refs[0])
        if self.language is None:
            # Fallback to en (this is only relevant for METEOR)
            self.language = 'en'

        self.filter = lambda s: s
        if filters:
            self.filter = FilterChain(filters)
            self.refs = self.filter(refs)

        assert len(self.refs) > 0, "Number of reference files == 0"

        for metric in sorted(beam_metrics):
            if metric is "Q2AVL" or metric is "Q2AVP":
                self.simmt_scorers[metric] = getattr(metrics, metric + 'Scorer')()
            else:
                self.kwargs[metric] = {'language': self.language}
                self.scorers[metric] = getattr(metrics, metric + 'Scorer')()

    def score(self, hyps, translator_outputs=None):
        """hyps is a list of hypotheses as they come out from decoder."""
        assert isinstance(hyps, list), "hyps should be a list."

        # Post-process if requested
        hyps = self.filter(hyps)

        results = []
        for key, scorer in self.scorers.items():
            results.append(
                scorer.compute(self.refs, hyps, **self.kwargs[key]))

        if len(translator_outputs) > 0:
            for key, scorer in self.simmt_scorers.items():
                # for the moment we take the first quality metric
                results.append(
                    scorer.compute(translator_outputs[0], results[0].score))
        return results
