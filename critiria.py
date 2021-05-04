import torch
import torch.nn as nn
import torch.nn.functional as F


class contrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(contrastiveLoss,self).__init__()
        self.margin = margin
        self.eps = 1e-4
    def forward(self, output1, output2, target):
        dist = F.pairwise_distance(output1, output2)
        loss = 0.5*(
            (1 - target.float()) * dist +
            target.float() * F.relu(self.margin-(dist+self.eps))
        )