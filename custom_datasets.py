import sys
import os
from skimage import io
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import random
import json
from PIL import Image


class datasetGen(Dataset):
    def __init__(self):
        f = open("JSONFiles/imgs_NamesLbls.json")  # img1 img2 img1_lbl img2_lbl
        train = json.load(f)
        self.img_path = train["img"]
        self.imgLbl_path = train["label"]
        self.num = len(self.img_path)

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        img = Image.open(self.img_path[index])
        img_lbl = self.imgLbl_path[index]
        img = self.transform(img)
        return img, img_lbl


# class siameseDataset(Dataset):
#     def __init__(self):
#         f = open("JSONFiles/imgs_pairsGen.json") #img1 img2 img1_lbl img2_lbl
#         train = json.load(f)
#         self.img1_path = train['img1']
#         self.img2_path = train['img2']
#         self.img1Lbl_path = train['img1_lbl']
#         self.img2Lbl_path = train['img2_lbl']
#         self.num = len(self.img1_path)

#         self.transform = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.Resize((256,256)),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ])

#     def __len__(self):
#         return self.num

#     def __getitem__(self,index):
#         img1 = Image.open(self.img1_path[index])
#         img2 = Image.open(self.img2_path[index])
#         img1_lbl = self.img1Lbl_path[index]
#         img2_lbl = self.img2Lbl_path[index]

#         img1 = self.transform(img1)
#         img2 = self.transform(img2)

#         if img1_lbl == img2_lbl:
#             target = 1
#         else:
#             target = 0
#         return (img1, img2), target


class siameseDataset(Dataset):
    def __init__(self):
        f = open("JSONFiles/imgs_NamesLbls.json")  # img1 img2 img1_lbl img2_lbl
        train = json.load(f)
        self.img_path = train["img"]
        self.imgLbl_path = train["label"]
        self.label_set = set(self.imgLbl_path)
        self.num = len(self.img_path)
        self.indices_to_labels = {
            label: np.where(np.array(self.imgLbl_path) == label)[0]
            for label in self.label_set
        }

    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        img1, label1 = Image.open(self.img_path[index]), self.imgLbl_path[index]
        if target == 1:
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.indices_to_labels[label1])
        else:
            siamese_label = np.random.choice(list(self.label_set - set([label1])))
            siamese_index = np.random.choice(self.indices_to_labels[siamese_label])

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        img2 = Image.open(self.img_path[siamese_index])
        img1, img2 = self.transform(img1), self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return self.num


class tripletDataset(Dataset):
    def __init__(self):
        f = open("JSONFiles/imgs_NamesLbls.json")  # img1 img2 img1_lbl img2_lbl
        train = json.load(f)
        self.img_path = train["img"]
        self.imgLbl_path = train["label"]
        self.label_set = set(self.imgLbl_path)
        self.num = len(self.img_path)
        self.indices_to_labels = {
            label: np.where(np.array(self.imgLbl_path) == label)[0]
            for label in self.label_set
        }

    def __getitem__(self, index):
        img1, label1 = Image.open(self.img_path[index]), self.imgLbl_path[index]
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.indices_to_labels[label1])
        negative_label = np.random.choice(list(self.label_set - set([label1])))
        negative_index = np.random.choice(self.indices_to_labels[negative_label])

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.Normalize([0.5, 0.5, 0.5], [0.4, 0.4, 0.4]),
            ]
        )

        img2 = Image.open(self.img_path[positive_index])
        img3 = Image.open(self.img_path[negative_index])
        img1, img2, img3 = (
            self.transform(img1),
            self.transform(img2),
            self.transform(img3),
        )
        return (img1, img2, img3), []

    def __len__(self):
        return self.num
