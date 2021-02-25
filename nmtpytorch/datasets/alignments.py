from pathlib import Path

import torch
from torch.utils.data import Dataset

import numpy as np
import json


class AlignmentsDataset(Dataset):

    NO_REGION = -1

    def __init__(self, fname, revert=False, is_json=True, **kwargs):
        self.path = Path(fname)
        if not self.path.exists():
            raise RuntimeError('{} does not exist.'.format(self.path))
        if is_json:
            with open(self.path) as file:
                data_dictionary = json.load(file)
                self.data = self._read_alignments(data_dictionary)
        else:
            self.data = np.load(self.path)

        self.order = list(range(len(self.data)))

        if revert:
            self.order = self.order[::-1]

        # Dataset size
        self.size = len(self.order)

    @staticmethod
    def to_torch(batch, **kwargs):
        """Assumes x.shape == (n, *)."""
        max_cols = max([len(row) for element in batch for row in element])
        max_rows = max([len(element) for element in batch])
        # If the alignments did not include the eos token and we need it we can increase the row size to contain -1
        # for the eos token as well. We can eventually add a flag to disable this.
        max_rows += 1

        padded = [batch + [[AlignmentsDataset.NO_REGION] * (max_cols)] * (max_rows - len(batch)) for batch in batch]
        padded = torch.tensor([row + [AlignmentsDataset.NO_REGION] * (max_cols - len(row)) for element in padded for row in element])
        x = padded.view(-1, max_rows, max_cols)
        return x.permute(1, 0, 2)

    @staticmethod
    def _read_alignments(dictionary):
        return [value for _, value in dictionary.items()]

    def __getitem__(self, idx):
        return self.data[self.order[idx]]

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} '{}' ({} samples)\n".format(
            self.__class__.__name__, self.path.name, self.__len__())
        return s