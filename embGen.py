import time, sys, os
from itertools import product
from collections import OrderedDict, namedtuple
import logging

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision as tv
import torchvision.models as models
import torchvision.datasets as dset
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms as transforms
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from custom_datasets import siameseDataset, tripletDataset, datasetGen
from network import (
    VGGEmbNet,
    SiameseNet,
    TripletNet,
    AlexEmbNet,
    SQEEmbNet,
    ResnetEmbNet,
    EffNetB0EmbNet,
    EffNetB3EmbNet,
)

import PIL
from PIL import Image

from losses import ContrastiveLoss, TripletLoss

from train import RunBuilder, RunManager

params = OrderedDict(
    batch_size=[10],
    model=["tripletNet", "siameseNet"],  # Best: To be determined
    network=["alex","effB3", "resnet", "sqe"],  # Best: effB0 "effB0",, 
    margin=[1.0],  # Best: 1.
    opt=["adam"],  # Best: Adam
    lr=[0.001, 0.0001],  # Best: , 0.0001
    # lpips_like = [True, False], ####### NEXT STEP ######
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def embeddings_Gen():
    with torch.no_grad():
        m = RunManager()
        for run in RunBuilder.get_runs(params):
            m.save_embs_runs(run)
            if run.model == "tripletNet":
                if run.network == "sqe":
                    embedding_net = SQEEmbNet()
                    embedding_net.load_state_dict(
                        torch.hub.load_state_dict_from_url(
                            "https://github.com/syedfquadri/wgz/releases/download/v0.0-alpha.1/trip_sqe_0001.pth.tar", progress= True
                        )["state_dict"]
                    )
                elif run.network == "alex":
                    embedding_net = AlexEmbNet()
                    embedding_net.load_state_dict(
                        torch.hub.load_state_dict_from_url(
                            "https://github.com/syedfquadri/wgz/releases/download/v0.1-alpha.1/trip_alex_0001.pth.tar", progress= True
                        )["state_dict"]
                    )
                elif run.network == "resnet":
                    embedding_net = ResnetEmbNet()
                    embedding_net.load_state_dict(
                        torch.hub.load_state_dict_from_url(
                            "https://github.com/syedfquadri/wgz/releases/download/v0.3-alpha.1/trip_resnet_0001.pth.tar", progress= True
                        )["state_dict"]
                    )
                elif run.network == "effB3":
                    embedding_net = EffNetB3EmbNet()
                    embedding_net.load_state_dict(
                        torch.hub.load_state_dict_from_url(
                            "https://github.com/syedfquadri/wgz/releases/download/v0.2-alpha.0/trip_effB3_0001.pth.tar", progress= True
                        )["state_dict"]
                    )
            elif run.model == "siameseNet":
                if run.network == "sqe":
                    embedding_net = SQEEmbNet()
                    embedding_net.load_state_dict(
                        torch.hub.load_state_dict_from_url(
                            "https://github.com/syedfquadri/wgz/releases/download/v0.0-alpha.0/siam_sqe_00001.pth.tar", progress= True
                        )["state_dict"]
                    )
                elif run.network == "alex":
                    embedding_net = AlexEmbNet()
                    embedding_net.load_state_dict(
                        torch.hub.load_state_dict_from_url(
                            "https://github.com/syedfquadri/wgz/releases/download/v0.1-alpha.0/siam_alex_00001.pth.tar", progress= True
                        )["state_dict"]
                    )
                elif run.network == "resnet":
                    embedding_net = ResnetEmbNet()
                    embedding_net.load_state_dict(
                        torch.hub.load_state_dict_from_url(
                            "https://github.com/syedfquadri/wgz/releases/download/v0.3-alpha.0/siam_resnet_00001.pth.tar", progress= True
                        )["state_dict"]
                    )
                elif run.network == "effB3":
                    embedding_net = EffNetB3EmbNet()
                    embedding_net.load_state_dict(
                        torch.hub.load_state_dict_from_url(
                            "https://github.com/syedfquadri/wgz/releases/download/v0.2-alpha.1/siam_effB3_00001.pth.tar", progress= True
                        )["state_dict"]
                    )
            embedding_net = embedding_net.to(device)
            Dset = datasetGen()
            data_loader = DataLoader(Dset, batch_size=len(Dset), shuffle=True)
            m.begin_run(run, embedding_net)
            plt.figure(figsize=(10, 10))
            batch = next(iter(data_loader))
            img_embs = []
            labels = []
            for i in range(len(batch[1])):
                embedding_net.eval()
                img = batch[0][i].to(device)
                label = batch[1][i]
                img_emb = embedding_net(img.unsqueeze(0))
                img_embs.append(img_emb)
                labels.append(label)
            img_embs = torch.stack(img_embs).reshape(-1, 32)
            m.tb.add_embedding(img_embs, labels, batch[0])
            m.tb.close()

if __name__ == "__main__":
    embeddings_Gen()