import torch
from torch.utils.data import IterableDataset
import random

class BatchFileLoader(IterableDataset):
    def __init__(self, dataset, batch_size=100, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        for i, idx in enumerate(self.indices):
            for bx in self.dataset.get_batch(idx).split(self.batch_size, dim=1):
                yield bx

        # Clear the CUDA cache every N batches
        if (i + 1) % 10 == 0:  # Adjust the frequency as needed
            torch.cuda.empty_cache()

    def __len__(self):
        return len(self.dataset)        