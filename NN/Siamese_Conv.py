import torch
from torch import nn
import scipy.spatial
from torchsummary import summary
import torch.nn.functional as F
import torch


class ContrastiveLoss(nn.Module):

    def __init__(self, margin = 2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        # LABEL == 0 if the pair is positive, else 0
        pos = (1-label) * pow(distance, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )

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
           #nn.MaxPool2d(kernel_size=2),
           #nn.Dropout2d(p=0.3),

           #nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3)),
           #nn.ReLU(inplace=True),
           )

       # dense layers
       self.fc = nn.Sequential(

       nn.Flatten(),

       nn.Linear(in_features = 64*17*25, out_features = 10, bias = True),
       #nn.ReLU(inplace=True),

       #nn.Linear(in_features=64, out_features=10, bias=True),
       nn.Sigmoid()
       )

    def forward_once(self, inp):
        # forward to convolutional layer
        output = self.cnn(inp)
        # forward to fully connected layer
        output = self.fc(output)
        return output


    def forward(self, inp1, inp2):
        left = self.forward_once(inp1)
        right = self.forward_once(inp2)
        # default is euclidian
        distance = F.pairwise_distance(left, right)
        return distance







