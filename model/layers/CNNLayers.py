#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/11 18:38
# @Version : 1.0
# @File    : CNNLayers.py

import torch.nn as nn
import torch.nn.functional as F
import torch

class BasicLeNet(nn.Module):
    def __init__(self, in_dim=16 * 5 * 5, num_class=2, dropout=0.5):
        super(BasicLeNet, self).__init__()
        # self.fc1 = nn.Linear(in_dim, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, num_class)

    def read_out(self, out):
        out = out.view(out.size(0), -1)
        # print(out.size())
        # print(out.size())
        # out = self.dropout(self.fc1(out))
        # print(out.size())
        out = F.relu(self.dropout(self.fc1(out)))
        out = self.fc2(out)
        return out


'''
in_dim is : 32 x 32
'''


class LeNet1(BasicLeNet):
    def __init__(self, size_c, **kwargs):
        super(LeNet1, self).__init__(in_dim=18 * 5 * 5, **kwargs)
        self.conv1 = nn.Conv2d(size_c, 16, 5)
        self.conv2 = nn.Conv2d(16, 18, 5)

    def reset_parameters(self):
        all_res = [self.conv1, self.conv2, self.fc1, self.fc2]
        for res in all_res:
            res.reset_parameters()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        # out1 = out.detach()
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        # out2 = out.detach()
        out = F.max_pool2d(out, 2)
        return self.read_out(out)
