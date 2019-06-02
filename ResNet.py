# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Machine Learning: ResNet structure definition.
    The codes greatly refer to WuJie1010 's work.(https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.CConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, input):
        out = functional.relu(self.bn1(self.conv1(input)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(input)
        out = functional.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, biad=False)
        self.bn1 = nn.BatchNorm2d(64)
        ml = self._make_layer
        self.layer1 = ml(block, 64, num_blocks[0], stride=1)
        self.layer2 = ml(block, 128, num_blocks[1], stride=2)
        self.layer3 = ml(block, 256, num_blocks[2], stride=2)
        self.layer4 = ml(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        out = functional.relu(self.bn1(self.conv1))
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            out = layer(out)
        out = functional.avg_poo2d(out, 4)
        out = out.view(out.size(0), -1)
        out = functional.dropout(out, p=0.5, training=self.training)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

