#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: voxnet.py
Created: 2020-01-21 21:32:40
Author : Yangmaonan
Email : 59786677@qq.com
Description: VoxNet 网络结构
'''

import torch
import torch.nn as nn
from collections import OrderedDict


class VoxNet(nn.Module):
    def __init__(self, n_classes=10, input_shape=(32, 32, 32)):
        super(VoxNet, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.feat = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                         out_channels=32, kernel_size=5, stride=2)),
            ('relu1', torch.nn.ReLU()),
            ('drop1', torch.nn.Dropout(p=0.2)),
            ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=128, kernel_size=3)),
            ('relu2', torch.nn.ReLU()),
            ('pool2', torch.nn.MaxPool3d(2)),
            ('drop2', torch.nn.Dropout(p=0.3))
        ]))
        x = self.feat(torch.autograd.Variable(torch.rand((1, 1) + input_shape)))
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n

        self.mlp = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(dim_feat, 128)),
            ('relu1', torch.nn.ReLU()),
            ('drop3', torch.nn.Dropout(p=0.4)),
            ('fc2', torch.nn.Linear(128, self.n_classes))
        ]))

    def forward(self, x):
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


class VoxNetSegmentation(nn.Module):
    def __init__(self, n_classes=10, input_shape=(32, 32, 32)):
        super(VoxNetSegmentation, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.feat = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                         out_channels=32, kernel_size=5, stride=2)),
            ('relu1', torch.nn.ReLU()),
            #('drop1', torch.nn.Dropout(p=0.2)),
            ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=128, kernel_size=3, stride=2)),
            ('relu2', torch.nn.ReLU())
            #('drop2', torch.nn.Dropout(p=0.3))
        ]))
        x = self.feat(torch.autograd.Variable(torch.rand((1, 1) + input_shape)))
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n

        self.up1 = torch.nn.ConvTranspose3d(128, 64, 3, 2)
        self.up2 = torch.nn.ConvTranspose3d(64, 32, 3, 2)
        self.up3 = torch.nn.ConvTranspose3d(32, 1, 6, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.feat(x)

        print ("bottleneck")
        print (x.shape)

        x = self.up1(x)
        x = self.relu(x)

        print ("LEVEL 1 SHAPE")
        print (x.shape)

        x = self.up2(x)
        x = self.relu(x)

        print ("LEVEL 2 SHAPE")
        print (x.shape)

        x = self.up3(x)

        print ("OUTPUT SHAPE")
        print (x.shape)
        exit()

        
        x = torch.sigmoid(x)

        #print (x.shape)

        return x


class VoxNetSegmentationV2(nn.Module):
    def __init__(self, n_classes=10, input_shape=(32, 32, 32)):
        super(VoxNetSegmentationV2, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.feat = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1, out_channels=16, kernel_size=5, stride=2)),
            ('relu1', torch.nn.ReLU()),
            ('drop1', torch.nn.Dropout(p=0.2)),
            ('conv3d_2', torch.nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1)),
            ('relu2', torch.nn.ReLU()),
            ('drop2', torch.nn.Dropout(p=0.2)),
            ('conv3d_3', torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1)),
            ('relu3', torch.nn.ReLU()),
            ('drop3', torch.nn.Dropout(p=0.2))]))

        x = self.feat(torch.autograd.Variable(torch.rand((1, 1) + input_shape)))
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n

        self.up1 = torch.nn.ConvTranspose3d(64, 32, 3, 3)
        self.up2 = torch.nn.ConvTranspose3d(32, 16, 2, 1)
        self.up3 = torch.nn.ConvTranspose3d(16, 1, 2, 1)
        
        self.up4 = torch.nn.ConvTranspose3d(16, 1, 3, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.feat(x)

        x = self.up1(x)
        x = self.relu(x)
        x = self.up2(x)
        x = self.relu(x)
        x = self.up3(x)
        #x = self.relu(x)
        #x = self.up4(x)
        x = torch.sigmoid(x)

        #print (x.shape)

        return x

class VoxNetSkips(nn.Module):
    def __init__(self):
        super(VoxNetSkips, self).__init__()
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=2)

        # print (x.shape)
        # 6 6 6

        self.up1 = torch.nn.ConvTranspose3d(128, 32, 5, 2)
        self.up2 = torch.nn.ConvTranspose3d(64, 1, 4, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):

        #print ("INPUT SHAPE")
        #print (x.shape)

        x = self.conv1(x)
        x = self.relu(x)

        #print ("LEVEL 1")
        #print (x.shape)

        x_1 = x

        x = self.conv2(x)
        x = self.relu(x)

        #print ("LEVEL 2")
        #print (x.shape)

        x = self.conv3(x)
        x = self.relu(x)

        #print ("LEVEL 3")
        #print (x.shape)

        #x = self.feat(x)

        x = self.up1(x)
        x = self.relu(x)

        #print ("LEVEL 4")
        #print (x.shape)

        # Skip via addition
        x = torch.cat((x, x_1), 1)

        x = self.up2(x)

        #print ("LEVEL 5")
        #print (x.shape)

        x = torch.sigmoid(x)

        return x



if __name__ == "__main__":
    data = torch.rand([1, 1, 32, 32, 32]).cuda()
    voxnetV2 = VoxNetSegmentationV2().cuda()
    out = voxnetV2(data)
    print (out.shape)
