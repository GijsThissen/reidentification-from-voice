import torch
from torch import nn


class ContrastiveLoss(nn.Module):

    def __init__(self, margin = 2.0):
        super.(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        loss_contrastive = torch.mean((1.0-label) * torch.pow(distance, 2) + (label) *torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss_contrastive


class SiameseNetwork(nn.Module):

    def __init__(self):
       super(SiameseNetwork, self).__init__()
       self.cnn = nn.Sequential(
           nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3)),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2),

           nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3)),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2),
           nn.Dropout2d(p=0.3),

           nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3)),
           nn.ReLU(inplace=True),
           )

       # dense layers
       self.fc = nn.Sequential(

       nn.Flatten(),

       nn.Linear(in_features = 64*3*3, out_features = 64, bias = True),
       nn.ReLU(inplace=True),

       nn.Linear(in_features=64, out_features=10, bias=True),
       nn.Softmax(-1)
       )

    def forward_once(self, inp):
        # forward to convolutional layer
        output = self.cnn(inp)
        # forward to fully connected layer
        output = self.fc(output)
        return output


    def forward(self, inp1, inp2):
        out1 = self.forward_once(inp1)
        out2 = self.forward_once(inp2)
        return out1, out2
