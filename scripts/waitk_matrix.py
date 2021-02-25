#!/usr/bin/env python
import os
import sys
import glob
import argparse
from pathlib import Path
from collections import defaultdict
from hashlib import sha1

import numpy as np

import sacrebleu
import tabulate

from nmtpytorch.metrics.simnmt import AVPScorer, AVLScorer, CWMScorer, CWXScorer


"""This script should be run from within the parent folder where each nmtpy
experiment resides."""


SCORERS = [
    AVPScorer(add_trg_eos=False),
    AVLScorer(add_trg_eos=False),
    #CWMScorer(add_trg_eos=False),
    #CWXScorer(add_trg_eos=False),
]


def read_references(fname):
    lines = []
    with open(fname) as f:
        for line in f:
            lines.append(line.strip())
    return lines


def compute_bleu(fname, refs):
    hyps = open(fname).read()
    hashsum = sha1(hyps.encode('utf-8')).hexdigest()
    parent = fname.parent
    cached_bleu = parent / f'.{fname.name}__{hashsum}'
    if os.path.exists(cached_bleu):
        return float(open(cached_bleu).read().strip().split()[2])
    else:
        bleu = sacrebleu.corpus_bleu(
            hyps.strip().split('\n'), refs, tokenize='none')
        with open(cached_bleu, 'w') as f:
            f.write(bleu.format() + '\n')
        return float(bleu.format().split()[2])


def get_scores_per_folder(folder, test_set, refs, suffix=None):
    results = {}
    glob_pattern = f'{folder}/*.{test_set}'
    if suffix:
        glob_pattern += f'.{suffix}'

    # get hyp files
    hyps = glob.glob(glob_pattern + '.gs')

    bleus = np.array([compute_bleu(Path(hyp), refs) for hyp in hyps])
    results['BLEU'] = bleus.mean(), bleus.std()

    # get action seqs
    acts = glob.glob(glob_pattern + '.acts')
    for sc in SCORERS:
        if len(acts) == 0:
            # not available
            results[sc.name] = None
        else:
            scores = np.array([sc.compute_from_file(act).score for act in acts])
            results[sc.name] = scores.mean(), scores.std()

    q2_avp = results['BLEU'][0] / (results['AVP'][0] if len(acts) > 0 else 1)
    results['Q/AVP'] = q2_avp, 0

    formatted_results = {}
    for key, value in results.items():
        if value is not None:
            formatted_results[key] = f'{value[0]:4.2f} ({value[1]:.2f})'
        else:
            formatted_results[key] = 'N/A'

    key = folder
    if suffix:
        key += f'.{suffix}'

    return key, formatted_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='waitk-matrix',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Compute metrics for waitk runs",
        argument_default=argparse.SUPPRESS)

    parser.add_argument('-t', '--test-set', default='test_2016_flickr',
                        help='the test set.')
    parser.add_argument('-l', '--lang', required=True, type=str,
                        help='Target side language.')
    parser.add_argument('-k', '--k', default='1,2,3,5,7',
                        help='Comma splitted k values to analyse.')
    parser.add_argument(
        '-r', '--ref-root',
        default='/data/ozan/experiments/simnmt/nmtpy/data/multi30k',
        help='Folder where reference text files reside.')

    args = parser.parse_args()
    results = {}

    # tokenized reference file
    ref_file = f'{args.ref_root}/en-{args.lang}/{args.test_set}.lc.norm.tok.{args.lang}.dehyph'

    refs = [read_references(ref_file)]

    key, value = get_scores_per_folder(
        'snmt-rnn-unimodal', args.test_set, refs, '')
    results[key] = value

    key, value = get_scores_per_folder(
        'snmt-rnn-multimodal', args.test_set, refs, '')
    results[key] = value


    for k in args.k.split(','):
        key, value = get_scores_per_folder(
            'snmt-rnn-unimodal', args.test_set, refs, f'wait{k}')
        results[key] = value
        key, value = get_scores_per_folder(
            'snmt-rnn-multimodal', args.test_set, refs, f'wait{k}')
        results[key] = value

        key, value = get_scores_per_folder(
            f'wait{k}-rnn-unimodal', args.test_set, refs, f'wait{k}')
        results[key] = value
        key, value = get_scores_per_folder(
            f'wait{k}-rnn-multimodal', args.test_set, refs, f'wait{k}')
        results[key] = value


    headers = ['Name'] + list(list(results.values())[-1].keys())
    results = [[name, *[scores[key] for key in headers[1:]]] for name, scores in results.items()]
    # alphabetical sort
    results = sorted(results, key=lambda x: float(x[headers.index('BLEU')].split()[0]))
    # print
    print(tabulate.tabulate(results, headers=headers))
