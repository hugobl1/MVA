import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models
nclasses = 20
from torchsummary import summary

#Traditionnal approch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5,padding='same')
        self.batchN1=nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5,padding='same')
        self.batchN2=nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3,padding='same')
        self.batchN3=nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3,padding='same')
        self.batchN4=nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=2,padding='same')
        self.batchN5=nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.batchN1(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(self.batchN2(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.batchN3(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.batchN4(self.conv4(x)), 2))
        x = F.relu(F.max_pool2d(self.batchN5(self.conv5(x)), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p= 0.5)
        return self.fc3(x)


model=Net().cuda()
summary(model, (3,64,64))