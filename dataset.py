import sys, os, time

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import random

from PIL import Image, ImageFile
from config import CNF

class siameseDataset(Dataset):
    def __init__(self):
        data_root = "../input/shopee-product-matching"
        csv = pd.read_csv("../input/shopee-product-matching/train.csv")
        self.img_filename = csv['image']
        self.img_target = csv['label_group']
        self.img_path = os.path.join(data_root,self.img_filename)
        self.label_set = set(self.img_target)
        self.num = len(self.img_filename)
        self.indices_to_labels = {
            label: np.where(np.array(self.img_target)==label)[0]
            for label in self.label_set
        }

    def __len__(self):
        return self.num
    def __getitem__(self, index):
        target = np.random.randint(0,2)
        img1, lbl1 = Image.open(self.img_path[index]), self.img_target[index]
        if target == 0:
            siam_index = index
            while siam_index == index:
                siam_index = np.random.choice(self.indices_to_labels[lbl1])
        else:
            siam_lbl = np.random.choice(list(self.label_set-set([lbl1])))
            siam_index = np.random.choice(self.indices_to_labels[siam_lbl])
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img2 = Image.open(self.img_path[siam_index])
        img1, img2 = self.transform(img1), self.transform(img2)
        return (img1, img2), target
