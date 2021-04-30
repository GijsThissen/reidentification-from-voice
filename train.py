import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import utils.Data_Loader as DL
import torchaudio
import os
import NN.Siamese_Conv as NN
from pytorch_metric_learning import losses

current_dir = os.getcwd()

# training_settings
batch_size = 80
n_epochs = 1

# Preprocessing function
mfcc = torchaudio.transforms.MFCC(sample_rate =  11025, log_mels=True)

# Data
data_path = "/home/jakob/data/raw_data.pkl"
train_data = DL.DataLD(data_path, mfcc)

# Loss
contrastive_loss = losses.ContrastiveLoss()

# Optimizer
learning_rate = 1e-3

def train(train_data, batch_size,
          preprocessing_function, loss_function,
          learning_rate, n_epochs = 1, deviceid = -1):

    model = NN.SiameseNetwork()

    if deviceid > -1:
        model.cuda()

    optimzer = optim.Adam(model.parameters(), lr= learning_rate)

   updates = 0
   loss_history = []
   best_dev_pearson = -1.0
   best_epoch = -1

   for epoch in range(n_epochs):

       print(f"EPOCH {epoch}")
       loss = train_epoch( model, train_data,
                           loss_function, optimzer,
                           batch_size, epoch, deviceid)
       loss_history.append(loss)

       #EVAL HERE

def train_epoch( model, train_data,
                 loss_function, optimzer,
                 batch_size, epoch, deviceid):
    model.train()
    avg

    avg_loss = 0.0
    avg_acc = 0.0
    total = 0
    total_acc = 0.0
    total_loss = 0.0


    data_loader = train_data.get_loader(shuf= True, batch_size = batch_size)

    for iter, data in enumerate(data_loader):
        rec0, rec1, lbl = data

        if deviceid > -1:
            rec0 = autograd.(rec0.cud)
            rec1 = autograd.Variable(rec1.cuda())
        model.zero_grad()
        model.batch_size = len(train_labels)
        model.hidden = model.init_hidden()
        output = model(rec0.t()










# for i, data in enumerate(train_data.get_loader(shuf = True, batch_size = 10)):

#     print(f"=============\n==============\ntensor: {i}\n============")
#
#     d = SN.forward(rec0, rec1)
#     print(d.shape)
# #    print("euclidian distance between two vectors:", d)
#     print()
#     print(lbl)
#     #print(f"rec0: {rec0}\n \n rec1: {rec1} \n \n lbl: {lbl}")
#     if i > 20:
#         break
