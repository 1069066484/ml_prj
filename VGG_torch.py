# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Data Science: Some transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
from enum import IntEnum

class VGGType(IntEnum):
    VGG11 = 0
    VGG13 = 1
    VGG16 = 2
    VGG19 = 3


VGG_structures = [
    [64, None, 128, None, 256, 256, None, 512, 512, None, 512, 512, None],
    [64, 64, None, 128, 128, None, 256, 256, None, 512, 512, None, 512, 512, None],
    [64, 64, None, 128, 128, None, 256, 256, 256, None, 512, 512, 512, None, 512, 512, 512, None],
    [64, 64, None, 128, 128, None, 256, 256, 256, 256, None, 512, 512, 512, 512, None, 512, 512, 512, 512, None],
    ]


class VGG(nn.Module):
    def __init__(self, vgg_type):
        super(VGG, self).__init__()
        self.features = self._make_layers(VGG_structures[vgg_type])
        self.classifier = nn.Linear(512,7)

    def forward(self, input):
        out = self.features(input)
        #torch.Size([64, 512, 1, 1])
        #print(out.size())

        out = out.view(out.size(0), -1)
        out = functional.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, structure):
        layers = []
        in_channels = 3
        for out_ch in structure:
            layers += \
                [nn.MaxPool2d(kernel_size=2, stride=2)] if out_ch is None \
                else [nn.Conv2d(in_channels, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
            if out_ch is not None:
                in_channels = out_ch
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
                


