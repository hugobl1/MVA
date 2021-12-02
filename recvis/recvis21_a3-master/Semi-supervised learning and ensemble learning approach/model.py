import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
nclasses = 20

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net1 = models.resnext101_32x8d(pretrained=True)

        for param in self.net1.parameters():
            param.requires_grad = True
        num_ftrs1 = self.net1.fc.in_features
        self.net1.fc = nn.Sequential(
            nn.Flatten(),
        )
        self.net2= models.efficientnet_b7(pretrained=True)
        self.net3=models.regnet_x_32gf(pretrained=True)
        for param in self.net2.parameters():
            param.requires_grad = False
        for param in self.net3.parameters():
            param.requires_grad = False

        num_ftrs2 = self.net2.classifier[1].in_features
        self.net2.classifier[1] = nn.Sequential(
            nn.Flatten(),
        )
        num_ftrs3 = self.net3.fc.in_features
        self.net3.fc = nn.Sequential(
            nn.Flatten(),
        )
        self.classifier=nn.Sequential(
            nn.Linear(num_ftrs1+num_ftrs2+num_ftrs3 , 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256,20),
        )



    def forward(self, x):
        x1=self.net1(x)
        x2=self.net2(x)
        x3=self.net3(x)
        xlast=torch.cat((x1,x2,x3), dim=1)
        return self.classifier(xlast)

