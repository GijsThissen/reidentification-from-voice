import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import utils.Data_Loader as DL
import torchaudio
import os
import NN.Siamese_Conv as NN

# training_settings
batch_size = 1000
n_epochs = 4

# Preprocessing function
mfcc = torchaudio.transforms.MFCC(sample_rate =  11025, log_mels=True)

# Data

train_data_path = "/home/jakob/src/reidentification-from-voice/data/train_data.pkl"
val_data_path = "/home/jakob/src/reidentification-from-voice/data/val_data.pkl"

# Loss
contrastive_loss = NN.ContrastiveLoss()

# Optimizer
learning_rate = 1e-3

def train(train_data, batch_size,
          preprocessing_function, loss_function,
          learning_rate, n_epochs = 1, deviceid = -1):

    model = NN.SiameseNetwork(  )
    # conv1_stride = 4, n_embedding_nodes = 100,


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
        performance = evalute( model, val_data, deviceid)
        print(f"dev MAE: {performance}")
        loss_history.append(loss)

        #EVAL HERE



def train_epoch( model, train_data,
                 loss_function, optimzer,
                 batch_size, epoch, deviceid):
    model.train()


    avg_loss = 0.0
    avg_acc = 0.0
    total = 0
    total_acc = 0.0
    total_loss = 0.0


    data_loader = train_data.get_loader(shuf= True, batch_size = batch_size)

    for iter, data in enumerate(data_loader):

        rec0, rec1, lbl = data
        #print(len(data_loader))

        if deviceid > -1:
            rec0.requires_grad_().cuda()
            rec1.requires_grad_().cuda()
            lbl.cuda()
        else:
            rec0.requires_grad_()
            rec1.requires_grad_()
        model.zero_grad()
        model.batch_size = len(lbl)
        sim = model(rec0, rec1)
        loss = loss_function(sim , lbl.float().requires_grad_())
        loss.backward()
        optimzer.step()

        print(f"Loss {loss.item()}")

    return loss.item




def evaluate(model, val_data, deviceid = -1):

    model.eval()
    dataloader = val_data.get_loader(batch_size = 1)
    num_batches =  len(dataloader)

    for iter, data in enumerate(data_loader):

        rec0, rec1, lbl = data
        #print(len(data_loader))

        if deviceid > -1:
            rec0.requires_grad_().cuda()
            rec1.requires_grad_().cuda()
            lbl.cuda()
        else:
            rec0.requires_grad_()
            rec1.requires_grad_()
        model.zero_grad()
        model.batch_size = len(lbl)
        sim = model(rec0, rec1)
        error = abs(lbl.detach() - sim.detach())
        accum_error += error

    return accum_error / num_batches





def main():

    current_dir = os.getcwd()
    train_data = DL.DataLD(train_data_path, mfcc)
    val_data =  DL.DataLD(val_data_path, mfcc)

    train(train_data, batch_size, mfcc, contrastive_loss, learning_rate, n_epochs)
if __name__  == "__main__":
    main()
