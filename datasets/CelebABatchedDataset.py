import os
import torch
from torch.utils.data import Dataset
from collections import OrderedDict

class CelebABatchedDataset(Dataset):
    def __init__(self, batch_dir, indices, transform=None,cache_size=5):
        self.indices = indices
        self.batch_dir = batch_dir
        self.transform = transform
        self.batch_files = [
            os.path.join(batch_dir, f"celeba_batch_{i:03d}.pt") 
            for i in self.indices
            if os.path.exists(os.path.join(batch_dir, f"celeba_batch_{i:03d}.pt"))
        ]

        self.cache_size = cache_size
        self.cache = OrderedDict()  # file_path -> tensor
        _batch = torch.load(os.path.join(batch_dir, f"celeba_batch_{0:03d}.pt"), map_location='cpu', weights_only=False)
        self.batch_size = _batch.shape[0]

    def __len__(self):
        return len(self.indices)

    def _load_batch(self, file_path):
        if file_path in self.cache:
            self.cache.move_to_end(file_path)
            return self.cache[file_path]
        else:
            batch = torch.load(file_path, map_location='cpu', weights_only=False)
            self.cache[file_path] = batch
            self.cache.move_to_end(file_path)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
            return batch
    
    def get_batch(self, b_idx):
        if not b_idx in self.indices:
            assert Exception(f"Index {b_idx} not in this dataset, only present are {self.indices}!")
        file_path = os.path.join(self.batch_dir, f"celeba_batch_{b_idx:03d}.pt")
        return self._load_batch(file_path)
    
    def __getitem__(self, idx):
        b_idx, img_idx = divmod(idx, self.batch_size)
        batch = self.get_batch(b_idx)
        img = batch[img_idx]
        if self.transform:
            img = self.transform(img)
        return img