from torch.utils.data import Dataset
import numpy as np
import subprocess
import csv
import os
  
class UnsupervisedDataset(Dataset):
    """ 
    Wraps another dataset to sample from. Returns the sampled indices during iteration.
    In other words, instead of producing (X, y) it produces (X, y, idx)
    """
    def __init__(self, base_dataset, transform=None, labeling=False):
        self.base = base_dataset
        self.transform = transform
        self.labeling = labeling

    def __len__(self):
        return len(self.base)
    

    def __getitem__(self, idx):
        datapoint = self.base[idx]
        if isinstance(datapoint, tuple):
            datapoint, label = datapoint
        if self.transform is not None:
            datapoint = self.transform(datapoint)
        return datapoint if not self.labeling else (datapoint,label)
    
    def get_by_label(self, target_label, max_count=-1):
        """ Returns all instances with a given label. """
        if not self.labeling:
            raise ValueError("Labeling is disabled for this dataset.")
        
        matching_data = []
        count = 0
        for idx in range(len(self.base)):
            datapoint, label = self.__getitem__(idx)
            if label != target_label:
                continue
            matching_data.append(datapoint)
            count+=1
            if count == max_count:
                break
        
        return matching_data
