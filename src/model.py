import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class Conv1DBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Conv1d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LinearBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.lin = nn.Linear(in_ch, out_ch)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.lin(x)))


class Tnet(nn.Module):
   def __init__(self, k = 3):
      super().__init__()

      self.k = k
      self.convmodule = nn.ModuleList([ Conv1DBNReLU(k, 64),
                                        Conv1DBNReLU(64, 128),
                                        Conv1DBNReLU(128, 1024)])
      self.linearmodule = nn.ModuleList([LinearBNReLU(1024, 512),
                                         LinearBNReLU(512, 256)])
      self.fc3 = nn.Linear(256, k*k)
      self.flatten = nn.Flatten(1)

   def forward(self, x):
       # shape (batch_size,3,n)
       batch_shape = x.size(0)
       for m in self.convmodule:
           x = m(x)
       pool = nn.MaxPool1d(x.size(-1))(x)# shape (batch_size,1024,1)
       flat =self.flatten(pool) # shape (batch_size,1024)
       for m in self.linearmodule:
           flat = m(flat)
       init = torch.eye(self.k, requires_grad=True).repeat(batch_shape,1,1)
       if flat.is_cuda:
           init=init.cuda()
       matrix = self.fc3(flat).view(-1,self.k,self.k) + init
       return matrix

class Transform(nn.Module):
   def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = Conv1DBNReLU(3, 64)
        self.conv2 = Conv1DBNReLU(64, 128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.conv3 = nn.Conv1d(128,1024,1)

   def forward(self, x):
       matrix3x3 = self.input_transform(x)
       x = torch.bmm(torch.transpose(x,1,2), matrix3x3).transpose(1,2)
       x = self.conv1(x)
       matrix64x64 = self.feature_transform(x)
       x = torch.bmm(torch.transpose(x,1,2), matrix64x64).transpose(1,2)
       x = self.conv2(x)
       x = self.bn3(self.conv3(x))
       x = nn.MaxPool1d(x.size(-1))(x)
       output = nn.Flatten(1)(x)
       return output, matrix3x3, matrix64x64


class PointNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.transform = Transform()
        self.fc1 = LinearBNReLU(1024, 512)
        self.fc2 = LinearBNReLU(512, 256)
        self.fc3 = nn.Linear(256, classes)
        
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x, matrix3x3, matrix64x64 = self.transform(x)
        x = self.fc1(x)
        x =self.fc2(x)
        output = self.fc3(x)
        return self.logsoftmax(output), matrix3x3, matrix64x64