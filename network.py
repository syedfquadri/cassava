import os, sys, time

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.models as models

from efficientnet_pytorch import EfficientNet


class effNet(nn.Module):
    def __init__(self):
        super(effNet, self).__init__()
        self.preNet = EfficientNet.from_pretrained('efficientnet-b4')
        from param in self.preNet.parameters():
            param.requires_grad = True
        self.fc = nn.Sequential()
    def forward(self,x):
        h = self.preNet.extract_features(x)
        return h

class siameseNet(nn.Module):
    def __init__(self, embNet):
        super(siameseNet, self).__init__()
        self.embNet = embNet
    
    def forward(self, x1, x2):
        output1 = self.embNet(x1)
        output2 = self.embNet(x2)
        return output1, output2