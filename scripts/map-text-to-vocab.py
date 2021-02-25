#!/usr/bin/env python
import sys

from nmtpytorch.vocabulary import Vocabulary


if __name__ == '__main__':
    try:
        inp_fname = sys.argv[1]
        vocab = sys.argv[2]
    except IndexError as exc:
        print(f'Usage: {sys.argv[0]} <input file> <vocab .json>')
        sys.exit(1)

    vocab = Vocabulary(vocab)

    with open(inp_fname) as f:
        for line in f:
            idxs = vocab.sent_to_idxs(line.strip())
            sent = vocab.idxs_to_sent(idxs)
            print(sent)
