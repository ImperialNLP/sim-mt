from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np


class EntitiesDataset(Dataset):

    def __init__(self, fname, revert=False, **kwargs):
        self.path = Path(fname)
        if not self.path.exists():
            raise RuntimeError('{} does not exist.'.format(self.path))

        self.data = np.load(self.path)
        self.order = list(range(self.data.shape[0]))

        if revert:
            self.order = self.order[::-1]

        # Dataset size
        self.size = len(self.order)

    @staticmethod
    def to_torch(batch, **kwargs):
        """Assumes x.shape == (n, *)."""
        x = torch.from_numpy(np.array(batch, dtype='float32'))
        return x.view(*x.size()[:2], -1).permute(2, 0, 1)

    def __getitem__(self, idx):
        return self.data[self.order[idx]]

    def __len__(self):
        return self.size

    def __repr__(self):
        s = "{} '{}' ({} samples)\n".format(
            self.__class__.__name__, self.path.name, self.__len__())
        return s
