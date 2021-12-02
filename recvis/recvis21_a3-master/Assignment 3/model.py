import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import timm
nclasses = 20


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net1=timm.create_model('vit_base_patch16_224', pretrained=True,
 num_classes=nclasses)
        self.net1.head = nn.Sequential(
            nn.Flatten(),
        )
        self.net2=models.resnext101_32x8d(pretrained=True)
        self.net2.fc=nn.Sequential(
          nn.Linear(2048,500),
        )
        for param in self.net1.parameters():
            param.requires_grad = True

        self.classifier=nn.Sequential(
            nn.Linear(768+500 , 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256,20),
        )

    def forward(self, x):
        x1=self.net1(x)
        x2=self.net2(x)
        xlast=torch.cat((x1,x2), dim=1)
        return self.classifier(xlast)
