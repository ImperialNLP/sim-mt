#!/usr/bin/env python
from collections import Counter
import random

import sys
import os
from pathlib import Path

import sacrebleu
from tabulate import tabulate


from nmtpytorch.metrics.simnmt import AVPScorer, AVLScorer


ORDER_IDX = {'min': 0, 'median': 1, 'max': -1}


def read_sentlevel_bleu(fname):
    scores = []
    with open(fname) as f:
        for line in f:
            scores.append(float(line.strip()))
    return scores


def read_sentlevel_meteor(fname):
    scores = []
    with open(fname) as f:
        for line in f:
            if line.startswith('Segment'):
                score = float(line.split()[-1])
                scores.append(score)
    return scores



def get_sentlevel_scores(prefixes, metric='bleu'):
    scorer = read_sentlevel_bleu if metric == 'bleu' else read_sentlevel_meteor
    runs = None
    acts = None

    for prefix in prefixes:
        hyps = open(prefix).read().strip().split('\n')
        actions = open(prefix.with_suffix('.acts')).read().strip().split('\n')
        sentscores = scorer(f'{prefix}.sent{metric}s')
        if runs is None:
            runs = [[] for i in range(len(hyps))]
            acts = [[] for i in range(len(hyps))]

        for idx, (score, hyp, act) in enumerate(zip(sentscores, hyps, actions)):
            runs[idx].append((score, hyp, act))

    sorted_runs = []
    for run in runs:
        sorted_runs.append(sorted(run, key=lambda x: x[0]))
    return sorted_runs


def reduce_runs(k_scores, ks, idx, method='median'):
    midx = ORDER_IDX.get(method, random.randint(0, 2))
    cands = [k_scores[k][idx][midx] + (k,) for k in ks]
    best = sorted(cands, key=lambda x: (x[0], -x[-1]))[-1]
    return best


def main(method, metric='bleu'):
    systems = [
        'snmt-rnn-unimodal',
        'wait{k}-rnn-unimodal',
        #'snmt-rnn-multimodal-decatt-bothlnorm',
        #'wait{k}-rnn-multimodal-decatt-bothlnorm',
        'snmt-rnn-multimodal-objdet-roi-bothlnorm',
        'wait{k}-rnn-multimodal-objdet-roi-bothlnorm',
        #'wait{k}-rnn-multimodal-objdet-roi-bothlnorm-warmup',
        'snmt-rnn-multimodal-san-roifeats-1head',
        'wait{k}-rnn-multimodal-san-roifeats-1head',
        #'snmt-rnn-multimodal-discreteattrsobjs-bothlnorm',
        #'wait{k}-rnn-multimodal-discreteattrsobjs-bothlnorm',
        #'wait{k}-rnn-multimodal-discreteattrsobjs-bothlnorm-warmup',
        #'snmt-rnn-multimodal-san-discreteattrsobjs-1head',
        #'wait{k}-rnn-multimodal-san-discreteattrsobjs-1head',
        #'wait{k}-rnn-multimodal-san-discreteattrsobjs-1head-warmup',
    ]

    #ks = [1, 2, 3, 4, 5, 6, 7]
    ks = [1, 2, 3]
    test_sets = ['test_2016_flickr', 'test_2017_flickr', 'test_2017_mscoco']
    delay_scorers = [
        #AVPScorer(add_trg_eos=False),
        AVLScorer(add_trg_eos=False),
    ]

    ofold = Path('ORACLES')
    ofold.mkdir(exist_ok=True, parents=True)

    for test_set in test_sets:
        refs = open(f'{test_set}.ref').readlines()
        print(f'{"-"*len(test_set)}\n{test_set}\n{"-"*len(test_set)}')
        results = []
        for system in systems:
            out_name = ofold / f'{system.replace("-", "_")}.{method}_{metric}.{test_set}.oracle'
            out_name = str(out_name).replace('wait{k}', 'trained')
            k_scores = {}
            winner_ks = []
            winner_hyps = []
            winner_acts = []

            for k in ks:
                files = list(Path(system.format(k=k)).glob(f'*.{test_set}.wait{k}.gs'))
                if len(files) > 0:
                    k_scores[k] = get_sentlevel_scores(files, metric)

            if not k_scores:
                continue

            n_sents = len(next(iter(k_scores.values())))

            of = open(out_name, 'w')
            for idx in range(n_sents):
                score, hyp, act, k = reduce_runs(
                    k_scores, k_scores.keys(), idx, method)
                winner_ks.append(k)
                winner_hyps.append(hyp)
                winner_acts.append(act)
                of.write(hyp + '\n')
            of.close()

            # Compute some stats
            ctr = Counter(winner_ks).most_common()
            denom = sum([c[1] for c in ctr])
            bleu = sacrebleu.corpus_bleu(
                winner_hyps, [refs], tokenize='none', lowercase=False)
            scores = {
                'name': system.replace('wait{k}', 'trained'),
                'BLEU': bleu.score,
            }

            for sc in delay_scorers:
                score = sc.compute(winner_acts)
                scores[score.name] = score.score

            ctr = dict(ctr)
            scores['DISTK'] = {k: ctr[k] for k in ks}
            results.append(scores)

        results = sorted(results, key=lambda x: x['BLEU'])
        print(tabulate(results, headers='keys', floatfmt='.2f', tablefmt='simple'))


if __name__ == '__main__':
    try:
        method = sys.argv[1]
    except:
        print(f'Usage: {sys.argv[0]} <max|median>')
        sys.exit(1)

    main(method)


