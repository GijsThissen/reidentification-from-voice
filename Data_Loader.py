import os
import pickle
import torch
import torch.autograd as autograd
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import itertools.combinations




class Reidentification_From_Voice(Dataset):
    
    def __init__(self, data_path):
        with open(data_path, 'rb') as handle:
            self.data = pickle.load(handle)

        indices = [ i for i in range(len(data[1]))]
        self.training_pairs = itertools.combinations_with_replacement(indices, 2)
        self.total_pairs = len(self.training_pairs)

    def __getitem__(self, idx):
        """Returns three Tensors: rec_1, rec_2 and label."""
        
        idx_0, idx_1  = self.training_pairs[idx]

        rec_0 = torch.from_numpy(data[0][idx_1])
        lbl_0 = data[1][idx_0]

        rec_1 = torch.from_numpy(data[0][idx_2])
        lbl_1 = data[1][idx_0]

        label = 1 if lbl_0 == lbl_1 else 0
        return rec_0, rec_1, torch.BoolTensor(label)

    def __len__(self):
        return self.total_pairs


class DataLD(object):

    def __init__(self, data_path, preprocessing_function):
        self.dataset = Reidentification_From_Voice(data_path)

    def get_loader(self, shuf= True, batch_size = 64):

        data_loader = DataLoader(dataset = self.dataset,
                                 batch_size = batch_size,
                                 shuffle=shuf)
        return data_loader
