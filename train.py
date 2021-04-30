import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import utils.Data_Loader as DL
import torchaudio
import os
import NN.Siamese_Conv as NN

current_dir = os.getcwd()

# training_settings
batch_size = 80
n_epochs = 1

# Preprocessing function
mfcc = torchaudio.transforms.MFCC(sample_rate =  11025, log_mels=True)

# Data
data_path = current_dir + "/data/raw_data.pkl"
train_data = DL.DataLD(data_path, mfcc)

# Loss
contrastive_loss = NN.ContrastiveLoss()


# Optimizer
learning_rate = 1e-3






SN = NN.SiameseNetwork()


print(SN)



for i, data in enumerate(train_data.get_loader(shuf = True, batch_size = 1)):

     print(f"=============\n==============\ntensor: {i}\n============")
     rec_0, id_0, rec_1, id_1, label = data
     d = SN.forward(rec_0, rec_1)
     #print(d.shape)
     print("euclidian distance between two vectors:", d)
     print()
     #print(lbl)
     #print(f"rec0: {rec0}\n \n rec1: {rec1} \n \n lbl: {lbl}")
     if i > 20:
         break
