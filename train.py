import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn as nn
import utils.Data_Loader as DL
import torchaudio
import os
import NN.Siamese_Conv as NN
import time

# training_settings
batch_size = 64
n_epochs = 4

# Preprocessing function
mfcc = torchaudio.transforms.MFCC(sample_rate =  11025, log_mels=True)

# Data

train_data_path = "/home/scrappy/src/reidentification-from-voice/data/train_data.pkl"
val_data_path = "/home/scrappy/src/reidentification-from-voice/data/val_data.pkl"

# Loss
contrastive_loss = NN.ContrastiveLoss()

# Optimizer
learning_rate = 1e-3

deviceid = 0
def train(train_data, val_data, batch_size,
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
        performance = evaluate( model, val_data, deviceid)
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
    # print(len(data_loader))

    # starttime = time.time()
    for iter, data in enumerate(data_loader):
        rec0, rec1, lbl = data
        #print(len(data_loader))

        if iter == 10:
            # endtime = time.time()
            # batch_time =  endtime - starttime
            # epoch_time = (batch_time/ 10) * 50000 / 60 / 60
            # print(epoch_time)
            break
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
    print(f"Evaluation")

    model.eval()
    data_loader = val_data.get_loader(batch_size = 1)
    num_batches =  len(data_loader)

    accum_error = 0
    for iter, data in enumerate(data_loader):

        rec0, rec1, lbl = data
        #print(len(data_loader))

        if iter == 1000:
            break


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
        print(lbl.detach())
        error = torch.nn.functional.binary_cross_entropy(sim.detach(), lbl.detach())
        accum_error += error

    return accum_error / 1000





def main():

    current_dir = os.getcwd()
    train_data = DL.DataLD(train_data_path, mfcc, deviceid)
    val_data =  DL.DataLD(val_data_path, mfcc, deviceid)

    train(train_data,val_data, batch_size, mfcc, contrastive_loss, learning_rate, n_epochs, deviceid)
if __name__  == "__main__":
    main()
