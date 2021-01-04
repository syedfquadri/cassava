import time, os, sys
# import logging
# from collections import namedtuple

import torch
import torch.nn as nn
# import torchvision as tv
import torchvision.models as models
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils import model_zoo
from torchvision import transforms as transforms

# import numpy as np
# import pandas as pd
import json
# import matplotlib.pyplot as pyplot

from network import (
    AlexEmbNet,
    SQEEmbNet,
    ResnetEmbNet,
    EffNetB3EmbNet,
)

import argparse

import PIL
from PIL import Image

# from pytorch_metric_learning.distances import *

# Declare
img1_path = ""
img2_path = ""

# Parser
parser = argparse.ArgumentParser(
    description="Inference through trained models using WGZ food images"
)
parser.add_argument(
    "--p1",
    "--img1_path",
    type=str,
    default="ex_imgs/a1.jpg",
    help="Enter path to Image 1. Default: ex_imgs/a1.jpg",
)
parser.add_argument(
    "--p2",
    "--img2_path",
    type=str,
    default="ex_imgs/b1.jpg",
    help="Enter path to Image 2. Default: ex_imgs/b1.jpg",
)
parser.add_argument(
    "--emb",
    "--embnet",
    type=str,
    default="sqe",
    help="Preferred type of trained emdeddings generator neural network. \nDefault: SqueezeNet[sqe].\nOptions: [alex,sqe,effB3,resnet]",
)
parser.add_argument(
    "--m",
    "--model",
    type=str,
    default="trip",
    help="Preferred model/architecture on which the chosen embedding network has been trained on. Default: Triplet[trip]. Options: [trip,siam]",
)
parser.add_argument("--use_gpu", action="store_true", help="turn on flag to use GPU")
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# logs_filename = "infer.log"
# logging.basicConfig(
#     filename=logs_filename,
#     level=logging.DEBUG,
#     format="%(asctime)s：%(msecs)d:%(levelname)s:%(message)s",
# )

if opt.m == "trip":
    if opt.emb == "sqe":
        embnet_path = "/v0.0-alpha.1/trip_sqe_0001.pth.tar"
        embNet = SQEEmbNet()
    elif opt.emb == "alex":
        embnet_path = "/v0.1-alpha.1/trip_alex_0001.pth.tar"
        embNet = AlexEmbNet()
    elif opt.emb == "effB3":
        embnet_path = "/v0.2-alpha.0/trip_effB3_0001.pth.tar"
        embNet = EffNetB3EmbNet()
    elif opt.emb == "resnet":
        embnet_path = "/v0.3-alpha.1/trip_resnet_0001.pth.tar"
        embNet = ResnetEmbNet()
elif opt.m == "siam":
    if opt.emb == "sqe":
        embnet_path = "/v0.0-alpha.0/siam_sqe_00001.pth.tar"
        embNet = SQEEmbNet()
    elif opt.emb == "alex":
        embnet_path = "/v0.1-alpha.0/siam_alex_00001.pth.tar"
        embNet = AlexEmbNet()
    elif opt.emb == "effB3":
        embnet_path = "/v0.2-alpha.1/siam_effB3_00001.pth.tar"
        embNet = EffNetB3EmbNet()
    elif opt.emb == "resnet":
        embnet_path = "/v0.3-alpha.0/siam_resnet_00001.pth.tar"
        embNet = ResnetEmbNet()

img1_path = opt.p1
img2_path = opt.p2
use_gpu = opt.use_gpu


def get_distance(img1_path, img2_path, embNet, embnet_path, use_gpu):
    with torch.no_grad():
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        img1 = transform(img1)
        img2 = transform(img2)
        if use_gpu == True:
            img1, img2 = img1.to(device), img2.to(device)
            embNet = embNet.to(device)
        embNet.eval()
        github_release_dir = "https://github.com/syedfquadri/wgz/releases/download"
        state_dict = github_release_dir + embnet_path
        embNet.load_state_dict(
            torch.hub.load_state_dict_from_url(state_dict, progress=True)["state_dict"]
        )
        infer_start = time.time()
        img1_emb, img2_emb = embNet.forward(img1.unsqueeze(0)), embNet.forward(
            img2.unsqueeze(0)
        )
        sig = nn.Sigmoid()
        sqrd_dist = (img2_emb - img1_emb).pow(2).sum(1)
        normed_sqrd_dist = sig(sqrd_dist)
        infer_time = time.time() - infer_start
        message = f"""Squared Distance between {img1_path} and {img2_path}: {sqrd_dist}.\nNormed Squared Similarity between {img1_path} and {img2_path}: {normed_sqrd_dist}.\nInference Time: {infer_time}."""
        print(message)


if __name__ == "__main__":
    get_distance(img1_path, img2_path, embNet, embnet_path, use_gpu)
