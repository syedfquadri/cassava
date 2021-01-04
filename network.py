import os, sys, time

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.models as models

from collections import namedtuple, OrderedDict
from efficientnet_pytorch import EfficientNet


class VGGEmbNet(nn.Module):
    def __init__(self):
        super(VGGEmbNet, self).__init__()
        self.preNet = models.vgg16(pretrained=True).features
        self.vgg = nn.Sequential()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.NSlices = 5
        for x in range(4):
            self.slice1.add_module(str(x), self.preNet[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), self.preNet[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), self.preNet[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), self.preNet[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), self.preNet[x])
        for param in self.preNet.parameters():
            param.requires_grad = False
        self.Final_FC = nn.Sequential()

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


# class ResNextEmbNet(nn.Module):
#     def __init__(self):
#         super(ResNextEmbNet, self).__init__()
#         self.preNet = models.resnext50_32x4d(True)
#         self.resNext = nn.Sequential()


class AlexEmbNet(nn.Module):
    def __init__(self):
        super(AlexEmbNet, self).__init__()
        self.pre_Net = models.alexnet(True).features

        # lpips like = False
        self.preNet = nn.Sequential()
        for x in range(13):
            self.preNet.add_module(str(x), self.pre_Net[x])
        for param in self.preNet.parameters():
            param.requires_grad = False

        # lpips like = True
        # self.slice1 = nn.Sequential()
        # self.slice2 = nn.Sequential()
        # self.slice3 = nn.Sequential()
        # self.slice4 = nn.Sequential()
        # self.slice5 = nn.Sequential()
        # self.NSlices = 5
        # for x in range(3):
        #     self.slice1.add_module(str(x),self.preNet[x])
        # for x in range(3,6):
        #     self.slice1.add_module(str(x),self.preNet[x])
        # for x in range(6,8):
        #     self.slice1.add_module(str(x),self.preNet[x])
        # for x in range(8,10):
        #     self.slice1.add_module(str(x),self.preNet[x])
        # for x in range(10,13):
        #     self.slice1.add_module(str(x),self.preNet[x])
        # for param in self.preNet.parameters():
        #     param.requires_grad = False
        # self.Final_FC = nn.Sequential()

        self.conv_block = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv2d(256, 256, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.Final_FC = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def forward(self, x):
        h = self.preNet(x)
        h = self.conv_block(h)
        h = h.reshape(-1, 256)
        h_out = self.Final_FC(h)
        return h_out


class SQEEmbNet(nn.Module):
    def __init__(self):
        super(SQEEmbNet, self).__init__()
        self.pre_Net = models.squeezenet1_1(pretrained=True).features

        # lpips like = False
        self.preNet = nn.Sequential()
        for x in range(13):
            self.preNet.add_module(str(x), self.pre_Net[x])
        for param in self.preNet.parameters():
            param.requires_grad = False

        # lpips like = True
        # self.slice1 = nn.Sequential()
        # self.slice2 = nn.Sequential()
        # self.slice3 = nn.Sequential()
        # self.slice4 = nn.Sequential()
        # self.slice5 = nn.Sequential()
        # self.slice6 = nn.Sequential()
        # self.slice7 = nn.Sequential()
        # self.NSlices = 5
        # for x in range(2):
        #     self.slice1.add_module(str(x), self.preNet[x])
        # for x in range(2,5):
        #     self.slice2.add_module(str(x), self.preNet[x])
        # for x in range(5, 8):
        #     self.slice3.add_module(str(x), self.preNet[x])
        # for x in range(8, 10):
        #     self.slice4.add_module(str(x), self.preNet[x])
        # for x in range(10, 11):
        #     self.slice5.add_module(str(x), self.preNet[x])
        # for x in range(11, 12):
        #     self.slice6.add_module(str(x), self.preNet[x])
        # for x in range(12, 13):
        #     self.slice7.add_module(str(x), self.preNet[x])
        # for param in self.preNet.parameters():
        #     param.requires_grad = False
        self.conv_block = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv2d(512, 512, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.Final_FC = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def forward(self, x):
        # lpips like = True
        # h = self.slice1(x)
        # h = self.slice2(h)
        # h = self.slice3(h)
        # h = self.slice4(h)
        # h = self.slice5(h)
        # h = self.slice6(h)
        # h = self.slice7(h)

        # lpips like False
        h = self.preNet(x)
        h_conv = self.conv_block(h)
        h_conv = h_conv.reshape(-1, 512)
        out = self.Final_FC(h_conv)
        return out


class EffNetB0EmbNet(nn.Module):
    def __init__(self):
        super(EffNetB0EmbNet, self).__init__()
        self.preNet = EfficientNet.from_pretrained("efficientnet-b0")
        for param in self.preNet.parameters():
            param.requires_grad = False
        self.Final_FC = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            # nn.Softmax(dim=1)
        )
        self.adptavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        h = self.preNet.extract_features(x)
        # h = self.sig(h)
        h = self.adptavg(h)
        h = h.reshape(-1, 1280)
        out = self.Final_FC(h)
        return out


class EffNetB3EmbNet(nn.Module):
    def __init__(self):
        super(EffNetB3EmbNet, self).__init__()
        self.preNet = EfficientNet.from_pretrained("efficientnet-b3")
        for param in self.preNet.parameters():
            param.requires_grad = False
        self.Final_FC = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            # nn.Softmax(dim=1)
        )
        self.adptavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        h = self.preNet.extract_features(x)
        # h = self.sig(h)
        h = self.adptavg(h)
        h = h.reshape(-1, 1536)
        out = self.Final_FC(h)
        return out


class ResnetEmbNet(nn.Module):
    def __init__(self):
        super(ResnetEmbNet, self).__init__()
        self.preNet = models.resnet50(pretrained=True)
        self.conv1 = self.preNet.conv1
        self.bn1 = self.preNet.bn1
        self.relu = self.preNet.relu
        self.maxpool = self.preNet.maxpool
        self.layer1 = self.preNet.layer1
        self.layer2 = self.preNet.layer2
        self.layer3 = self.preNet.layer3
        self.layer4 = self.preNet.layer4
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
        for param in self.layer3.parameters():
            param.requires_grad = False
        for param in self.layer4.parameters():
            param.requires_grad = False
        self.conv_block = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv2d(2048, 1024, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.sig = nn.Sigmoid()
        self.Final_FC = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.maxpool(h)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.conv_block(h)
        h_flat = h.reshape(-1, 1024)
        h_out = self.Final_FC(h_flat)
        return h_out


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        out1 = self.embedding_net(x1)
        out2 = self.embedding_net(x2)
        return out1, out2


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        out1 = self.embedding_net(x1)
        out2 = self.embedding_net(x2)
        out3 = self.embedding_net(x3)
        return out1, out2, out3
