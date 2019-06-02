# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Data Science: Some transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as functional






class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = self._make_layers_cov()
        self.classifier = self._make_layers_clf()

    def forward(self, input):
        out = self.features(input)
        out = out.view(out.size(0), -1)
        # print("out.size(0)=",out.size(0))
        # print("out.size(1)=",out.size(1))
        out = functional.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers_clf(self):
        return nn.Sequential(*[
            nn.Linear(16 * 11 * 11,144),
            nn.ReLU(inplace=True),
            nn.Linear(144,84),
            nn.ReLU(inplace=True),
            nn.Linear(84,7)])

    #[64 x 1936], m2: [2304 x 120]
    def _make_layers_cov(self):
        layers = []
        in_channels = 3
        # 48 -> 12 * 16
        LeNet5_structures = [
            6, None, 16, None
        ]
        structure = LeNet5_structures
        for out_ch in structure:
            layers += \
                [nn.MaxPool2d(kernel_size=2, stride=2)] if out_ch is None \
                else [nn.Conv2d(in_channels, out_ch, kernel_size=5, padding=2),
                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
            if out_ch is not None:
                in_channels = out_ch
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
