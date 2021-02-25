#!/usr/bin/env python
import numpy as np


def read_file(fname):
    lines = []
    with open(fname) as f:
        for line in f:
            lines.append(line.strip().split())
    return lines


if __name__ == '__main__':
    for lpair in ['en-de', 'en-fr']:
        sl, tl = lpair.split('-')
        for split in ['train', 'val', 'test_2016_flickr', 'test_2017_flickr', 'test_2017_mscoco']:
            src = f'{lpair}/{split}.lc.norm.tok.{sl}'
            trg = f'{lpair}/{split}.lc.norm.tok.{tl}'

            src_lines = read_file(src)
            trg_lines = read_file(trg)

            n_equal = 0
            n_trg_shorter = 0
            ratios = []
            for sline, tline in zip(src_lines, trg_lines):
                ratios.append(len(tline) / len(sline))
                if len(tline) < len(sline):
                    n_trg_shorter += 1
                elif len(tline) == len(sline):
                    n_equal += 1

            t2l_ratio = np.array(ratios).mean()
            print(f'{lpair} - {split} > target-to-source ratio: {t2l_ratio:.3f}')
            print(f'   > {n_trg_shorter} target sents shorter than source (ratio: {n_trg_shorter / len(src_lines): .2f})')
            print(f'   > {n_equal} sents are equal in lengths (ratio: {n_equal / len(src_lines): .2f})')
            print()

        print()
