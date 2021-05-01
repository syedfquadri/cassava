
# %% [code]
import os, time, sys
import math

# from __future__ import print_function


from tqdm.notebook import tqdm

import pandas as pd
import numpy as np

import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image

from efficientnet_pytorch import EfficientNet
# import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %% [code]
data_dir = "../input/shopee-product-matching/"
train_dir = "../input/shopee-product-matching/train_images/"
test_dir =  "../input/shopee-product-matching/test_images/"

train_df = pd.read_csv("../input/shopee-product-matching/train.csv")
test_df = pd.read_csv("../input/shopee-product-matching/test.csv")

# %% [code]
# Custom Datasets

class dataset(Dataset):
    def __init__(self, train=True):
        super(dataset, self).__init__()
        self.image_path = train_dir + train_df["image"]
        self.label = train_df["label_group"]
        self.ulabels = set(self.label)
        self.num = len(self.image_path)
        if train==True:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.25,0.25,0.25]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.25,0.25,0.25]),
            ])
    def __len__(self):
        return self.num
    def __getitem__(self, index):
        image = Image.open(self.image_path[index])
        image = self.transform(image)
        image_lbl = int(self.label[index])
        return image, image_lbl

class siameseDataset(Dataset):
    def __init__(self):
        super(siameseDataset, self).__init__()
        self.image_path = train_dir + train_df["image"]
        self.label = train_df['label_group']
        self.ulabels = set(self.label)
        self.num = len(self.image_path)
        train = True
        if train==True:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.25,0.25,0.25]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.25,0.25,0.25]),
            ])
        self.indices_to_labels = {
            label: np.where(np.array(self.label) == label)[0]
            for label in self.ulabels
        }
    def __len__(self):
        return self.num
    def __getitem__(self, index):
        target = np.random.randint(0,2)
        img1, lbl1 = Image.open(self.image_path[index]), self.label[index]
        if target == 1:
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.indices_to_labels[lbl1])
        else:
            siamese_label = np.random.choice(list(self.ulabels - set([lbl1])))
            siamese_index = np.random.choice(self.indices_to_labels[siamese_label])
        img2 = Image.open(self.image_path[siamese_index])
        img1, img2 = self.transform(img1), self.transform(img2)
        return (img1, img2), target

# %% [code]
# Pre-Proccessing


# %% [code]
# Embedding Networks (pretrained) Block
# VGG, Resnet, SE-VGG, BotNet, RepVGG, SWaV, Deit, LPIPS, SQE, FAISS, MobileNet

class effNet(nn.Module):
    def __init__(self):
        super(effNet,self).__init__()
        self.preNet = EfficientNet.from_pretrained("efficientnet-b4")
        for param in self.preNet.parameters():
            param.requires_grad = True
#         self.head = nn.sequential()
        self.adpAvg = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        h = self.preNet.extract_features(x)
        h = self.adpAvg(h)
        h = h.reshape(-1, 1792)
        return h

class SQEEmbNet(nn.Module):
    def __init__(self):
        super(SQEEmbNet, self).__init__()
        self.pre_Net = models.squeezenet1_1(pretrained=True).features
        self.preNet = nn.Sequential()
        for x in range(13):
            self.preNet.add_module(str(x), self.pre_Net[x])
        for param in self.preNet.parameters():
            param.requires_grad = True

        self.adpAvg = nn.AdaptiveAvgPool2d(1)
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
        h = self.preNet(x)
        h = self.adpAvg(h)
        h = h.reshape(-1, 512)
        # out = self.Final_FC(h_conv)
        return h

# %% [code]
# Siamese Net Block
class siameseNet(nn.Module):
    def __init__(self, embNet):
        super(siameseNet, self).__init__()
        self.embNet = embNet
    def forward(self, x1, x2):
        out1 = self.embNet(x1)
        out2 = self.embNet(x2)
        return out1, out2

# %% [code]
#Critiria Block


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



class contrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(contrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-4
    def forward(self, out1, out2, target, size_average=True):
        distances = (out2 - out1).pow(2).sum(1) #Squared Distances
        losses = 0.5*(target.float()*distances + (1 + -1 *target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean()

# %% [code]
# Metric

class f1Metric():
    def __init__(self):
        pass
    def __call__(self):
        pass

# %% [code]
# Train
batch_size = 4
margin = 1.0
lr = 0.0001

embeddingNet = effNet()
embeddingNet = embeddingNet.to(device)
def train_epoch(trainloader):
    train_loss = 0
    optimizer = optim.Adam(model.embNet.preNet.parameters(), lr = lr)
    print("inside train_epoch")
    for idx, data in enumerate(trainloader):
        img1 = data[0][0]
        img2 = data[0][1]
        target = data[1]
        img1, img2, target = img1.to(device), img2.to(device), target.to(device)
        optimizer.zero_grad()
        preds = model(img1, img2)
        loss = contrastiveLoss(margin)
        loss = loss(preds[0], preds[1], target)
        train_loss += loss
        loss.backward()
        optimizer.step()
        print(f'Training Batch: {idx}/{27400/15} | TrainLoss: {train_loss} | Loss: {loss}')
    return train_loss
def val_epoch(vallaoder):
    model.eval()
    val_loss = 0
    for idx, data in enumerate(valloader):
        img1 = data[0][0]
        img2 = data[0][1]
        target = data[1]
        # img1, img2, target = img1.to(device), img2.to(device), target.to(device)
        preds = model(img1, img2)
        loss = contrastiveLoss(margin)
        loss = loss(preds[0], preds[1], target)
        val_loss += loss
        print(f'Validation Batch: {idx}/{27400/15} | Total Val Loss: {train_loss} | Val Loss: {loss}')
    val_loss /= idx + 1
    return val_loss
for epoch in range(30):
    dset = siameseDataset()
    trainset, valset = torch.utils.data.random_split(dset, [27400, 6850])
    trainloader = DataLoader(trainset, batch_size, True)
    valloader = DataLoader(valset, batch_size, True)
    model = siameseNet(embeddingNet)
    trainLoss = train_epoch(trainloader)
    valLoss = val_epoch(valloader)
    print("="*80)
    print(f"epoch:{epoch} | train loss:{train_loss} | val loss:{val_loss}")
    print("#"*80)