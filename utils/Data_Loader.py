import pickle
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import itertools



class Reidentification_From_Voice(Dataset):
    def __init__(self, data_path, preprocessing_function, deviceid):
        with open(data_path, 'rb') as handle:
            self.data = pickle.load(handle)

        indices = [ i for i in range(len(self.data[1]))]
        self.training_pairs = list(itertools.combinations(indices, 2))
        print(len(self.training_pairs))
        self.total_pairs = len(self.training_pairs)
        self.deviceid = deviceid
        self.preprocessing_function = preprocessing_function



        # self.src_sents = open(src_filename).readlines()
        # self.trg_sents = open(trg_filename).readlines()
        # self.labels = open(labels_filename).readlines()

        # self.total_sents = len(self.labels)
        # self.bpe_model = spm.SentencePieceProcessor(model_file= bpe_model_path)

    def __getitem__(self, idx):
        """Returns three Tensors: rec_1, rec_2 and label."""


        idx_0, idx_1  = self.training_pairs[idx]
        lbl_0 = self.data[1][idx_0]
        lbl_1 = self.data[1][idx_1]
        if self.deviceid == -1:
            rec_0 = torch.from_numpy(self.data[0][idx_0])
            rec_1 = torch.from_numpy(self.data[0][idx_1])

            rec_0 = self.preprocessing_function.forward(rec_0)
            rec_1 = self.preprocessing_function.forward(rec_1)

            rec_0 = rec_0.reshape(1, rec_0.shape[0], rec_0.shape[1])
            rec_1 = rec_1.reshape(1, rec_1.shape[0], rec_1.shape[1])

            label = True if lbl_0 == lbl_1 else False
        else:

            rec_0 = torch.from_numpy(self.data[0][idx_0]).cuda()
            rec_1 = torch.from_numpy(self.data[0][idx_1]).cuda()

            rec_0 = self.preprocessing_function.forward(rec_0).cuda()
            rec_1 = self.preprocessing_function.forward(rec_1).cuda()

            rec_0 = rec_0.reshape(1, rec_0.shape[0], rec_0.shape[1]).cuda()
            rec_1 = rec_1.reshape(1, rec_1.shape[0], rec_1.shape[1]).cuda()

            label = True if lbl_0 == lbl_1 else False


        # print(rec_0.shape)
        # print(rec_1.shape)
        # print(lbl_0)
        # print(lbl_1)
        # print(label)
        # return rec_0, rec_1, label

    def __len__(self):
        return self.total_pairs


class DataLD(object):

    def __init__(self, data_path, preprocessing_function, deviceid = -1):
        self.dataset = Reidentification_From_Voice(data_path, preprocessing_function, deviceid)

    def get_loader(self, shuf= True, batch_size = 80):

        data_loader = DataLoader(dataset = self.dataset,
                                 batch_size = batch_size,
                                 shuffle=shuf)
        return data_loader
