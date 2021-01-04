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

# from pytorch_metric_learning import losses as losses
# from pytorch_metric_learning import miners
# import lpips

import PIL
from PIL import Image

from losses import ContrastiveLoss, TripletLoss

logs_filename = "results.log"
logging.basicConfig(
    filename=logs_filename,
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

##########################################################################################
# Tensorboard runs hosted here : https://tensorboard.dev/experiment/XaWjqJX0Sgamt37dAsrTMg
# NEW : https://tensorboard.dev/experiment/CgW5mHnqQMqBjT9Imo6iZQ/
##########################################################################################

params = OrderedDict(
    batch_size=[10],
    model=["tripletNet", "siameseNet"],  # Best: To be determined
    network=[
        "alex",
    ],  # Best: effB0 "effB0",, "effB3", "resnet", "sqe"
    margin=[1.0],  # Best: 1.
    opt=["adam"],  # Best: Adam
    lr=[0.001],  # Best: , 0.0001
    # lpips_like = [True, False], ####### NEXT STEP ######
)


class RunBuilder:
    @staticmethod
    def get_runs(params):
        Run = namedtuple("Run", params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs


class RunManager:
    def __init__(self):
        self.epoch_number = 0
        self.epoch_trainLoss = 0
        self.epoch_testLoss = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_number = 0
        self.run_data = []
        self.run_start_time = 0

        self.network = None
        self.tb = None

    def begin_run(self, run, embedding_net):
        self.run_start_time = time.time()

        self.run_params = run
        self.run_number += 1

        self.network = embedding_net
        self.tb = SummaryWriter(comment=f"-{run}")

    def end_run(self):
        self.tb.close()
        self.epoch_number = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_number += 1
        self.epoch_trainLoss = 0
        self.epoch_testLoss = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        trainLoss = self.epoch_trainLoss.item()
        testLoss = self.epoch_testLoss.item()

        self.tb.add_scalar("Train Loss:", trainLoss, self.epoch_number)
        self.tb.add_scalar("Test Loss:", testLoss, self.epoch_number)

        results = OrderedDict()
        results["run"] = self.run_number
        results["epoch"] = self.epoch_number
        results["train loss"] = trainLoss
        results["test loss"] = testLoss
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration
        for k, v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)

        message = f"{self.run_params}\n Run Data: {results}"
        logging.info(message)

    def track_trainLoss(self, train_Loss):
        self.epoch_trainLoss = train_Loss  # edittable

    def track_testLoss(self, test_Loss):
        self.epoch_testLoss = test_Loss  # edittable

    def save_model(self, state):
        print("Run Finished...\nSaving Model")
        model_path = "./models/"
        model_name = "_".join(str(v) for v in list(self.run_params))
        model_name = model_name.replace(".", "")
        model_name += "_Final_Model_999"
        model_name += ".pth.tar"
        modelname = os.path.join(model_path, model_name)
        torch.save(state, modelname, _use_new_zipfile_serialization=False)

    def save_score(self):
        pd.DataFrame.from_dict(self.run_data, orient="columns").to_csv(
            f"Final_results.csv"
        )

        with open(f"Final_results.json", "w", encoding="utf-8") as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)

    def save_embs_runs(self, run):
        self.tb = SummaryWriter(comment=f"-{run}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    m = RunManager()
    for run in RunBuilder.get_runs(params):
        if run.network == "sqe":
            embedding_net = SQEEmbNet()
        elif run.network == "alex":
            embedding_net = AlexEmbNet()
        elif run.network == "resnet":
            embedding_net = ResnetEmbNet()
        elif run.network == "effB0":
            embedding_net = EffNetB0EmbNet()
        elif run.network == "effB3":
            embedding_net = EffNetB3EmbNet()

        embedding_net = embedding_net.to(device)
        if run.model == "siameseNet":
            model = SiameseNet(embedding_net)
            Dset = siameseDataset()
            train_set, test_set = torch.utils.data.random_split(Dset, [100, 63])
            trainLoader = DataLoader(train_set, batch_size=run.batch_size, shuffle=True)
            testLoader = DataLoader(test_set, batch_size=run.batch_size, shuffle=True)
        elif run.model == "tripletNet":
            model = TripletNet(embedding_net)
            Dset = tripletDataset()
            train_set, test_set = torch.utils.data.random_split(Dset, [100, 63])
            trainLoader = DataLoader(train_set, batch_size=run.batch_size, shuffle=True)
            testLoader = DataLoader(test_set, batch_size=run.batch_size, shuffle=True)
        model = model.to(device)
        m.begin_run(run, embedding_net)

        def train_epoch(trainLoader):
            model.train()
            train_loss = 0
            if run.opt == "adam":
                if run.network == "effB0" or "effB3":
                    optimizer = optim.Adam(
                        [
                            # {
                            #     "params": model.embedding_net.preNet.parameters(),
                            #     "lr": run.lr,
                            # },
                            # {'params':model.embedding_net.conv_block.parameters(), 'lr':run.lr},
                            {
                                "params": model.embedding_net.Final_FC.parameters(),
                                "lr": run.lr,
                            },
                        ],
                        lr=run.lr,
                        weight_decay=0.0001,
                    )
                else:
                    optimizer = optim.Adam(
                        [
                            # {'params':model.embedding_net.preNet.parameters(),'lr':run.lr},
                            {
                                "params": model.embedding_net.conv_block.parameters(),
                                "lr": run.lr * 0.01,
                            },
                            {
                                "params": model.embedding_net.Final_FC.parameters(),
                                "lr": run.lr,
                            },
                        ],
                        lr=run.lr,
                        weight_decay=0.0001,
                    )
            # elif run.opt == "sgd":
            #     if run.network == "effB0":
            #         optimizer = optim.SGD(
            #             [
            #                 {
            #                     "params": model.embedding_net.preNet.parameters(),
            #                     "lr": run.lr,
            #                 },
            #                 # {'params':model.embedding_net.conv_block.parameters(), 'lr':run.lr},
            #                 {
            #                     "params": model.embedding_net.Final_FC.parameters(),
            #                     "lr": run.lr,
            #                 },
            #             ],
            #             lr=run.lr,
            #             momentum=0.9,
            #             weight_decay=0.0001,
            #         )
            #     else:
            #         optimizer = optim.SGD(
            #             [
            #                 # {'params':model.embedding_net.preNet.parameters(),'lr':run.lr},
            #                 {
            #                     "params": model.embedding_net.conv_block.parameters(),
            #                     "lr": run.lr * 0.01,
            #                 },
            #                 {
            #                     "params": model.embedding_net.Final_FC.parameters(),
            #                     "lr": run.lr,
            #                 },
            #             ],
            #             lr=run.lr,
            #             momentum=0.9,
            #             weight_decay=0.0001,
            #         )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", factor=0.01, threshold=1e-4
            )
            for batch_idx, data in enumerate(trainLoader):
                img1 = data[0][0].to(device)
                img2 = data[0][1].to(device)
                if run.model == "siameseNet":
                    target = data[1].to(device)
                    preds = model(img1, img2)
                    constLoss = ContrastiveLoss(run.margin)
                    loss_output = constLoss(preds[0], preds[1], target)
                elif run.model == "tripletNet":
                    img3 = data[0][2].to(device)
                    preds = model(img1, img2, img3)
                    tripLoss = TripletLoss(run.margin)
                    loss_output = tripLoss(preds[0], preds[1], preds[2])
                train_loss += loss_output
                optimizer.zero_grad()
                loss_output.backward()
                optimizer.step()
                scheduler.step(loss_output)
            train_loss /= batch_idx + 1
            return train_loss

        def test_epoch(testLoader):
            with torch.no_grad():
                model.eval()
                test_loss = 0
                for batch_idx, data in enumerate(testLoader):
                    img1 = data[0][0].to(device)
                    img2 = data[0][1].to(device)
                    if run.model == "siameseNet":
                        target = data[1].to(device)
                        preds = model(img1, img2)
                        constLoss = ContrastiveLoss(run.margin)
                        loss_output = constLoss(preds[0], preds[1], target)
                    elif run.model == "tripletNet":
                        img3 = data[0][2].to(device)
                        preds = model(img1, img2, img3)
                        tripLoss = TripletLoss(run.margin)
                        loss_output = tripLoss(preds[0], preds[1], preds[2])
                    test_loss += loss_output
                test_loss /= batch_idx + 1
                return test_loss

        for epoch in range(30):
            m.begin_epoch()
            train_loss = train_epoch(trainLoader)
            test_loss = test_epoch(testLoader)
            m.track_trainLoss(train_loss)
            m.track_testLoss(test_loss)
            # m.save_score()
            m.end_epoch()
        checkpoint = {"state_dict": model.embedding_net.state_dict()}
        m.save_model(checkpoint)
        m.end_run()


def embeddings_Gen():
    with torch.no_grad():
        m = RunManager()
        for run in RunBuilder.get_runs(params):
            if run.model == "tripletNet":
                if run.network == "sqe":
                    embedding_net = SQEEmbNet()
                    embedding_net.load_state_dict(
                        torch.load("./final_models/trip_sqe_0001.pth.tar")["state_dict"]
                    )
                elif run.network == "alex":
                    embedding_net = AlexEmbNet()
                    embedding_net.load_state_dict(
                        torch.load("./final_models/trip_alex_0001.pth.tar")[
                            "state_dict"
                        ]
                    )
                elif run.network == "resnet":
                    embedding_net = ResnetEmbNet()
                    embedding_net.load_state_dict(
                        torch.load("./final_models/trip_resnet_0001.pth.tar")[
                            "state_dict"
                        ]
                    )
                elif run.network == "effB0":
                    embedding_net = EffNetB0EmbNet()
                    embedding_net.load_state_dict(
                        torch.load("./final_models/trip_effB0_0001.pth.tar")[
                            "state_dict"
                        ]
                    )
                elif run.network == "effB3":
                    embedding_net = EffNetB3EmbNet()
                    embedding_net.load_state_dict(
                        torch.load("./final_models/trip_effB3_0001.pth.tar")[
                            "state_dict"
                        ]
                    )
            elif run.model == "siameseNet":
                if run.network == "sqe":
                    embedding_net = SQEEmbNet()
                    embedding_net.load_state_dict(
                        torch.load("./final_models/siam_sqe_00001.pth.tar")[
                            "state_dict"
                        ]
                    )
                elif run.network == "alex":
                    embedding_net = AlexEmbNet()
                    embedding_net.load_state_dict(
                        torch.load("./final_models/siam_alex_00001.pth.tar")[
                            "state_dict"
                        ]
                    )
                elif run.network == "resnet":
                    embedding_net = ResnetEmbNet()
                    embedding_net.load_state_dict(
                        torch.load("./final_models/siam_resnet_00001.pth.tar")[
                            "state_dict"
                        ]
                    )
                elif run.network == "effB0":
                    embedding_net = EffNetB0EmbNet()
                    embedding_net.load_state_dict(
                        torch.load("./final_models/siam_effB0_00001.pth.tar")[
                            "state_dict"
                        ]
                    )
                elif run.network == "effB3":
                    embedding_net = EffNetB3EmbNet()
                    embedding_net.load_state_dict(
                        torch.load("./final_models/siam_effB3_00001.pth.tar")[
                            "state_dict"
                        ]
                    )
            embedding_net = embedding_net.to(device)
            Dset = datasetGen()
            data_loader = DataLoader(Dset, batch_size=len(Dset), shuffle=True)
            m.begin_run(run, embedding_net)
            plt.figure(figsize=(9, 9))
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
            m.end_run()


def load_models(modelname):
    print("Loading Checkout/Saved Model...")
    pass


if __name__ == "__main__":
    # main()
    embeddings_Gen()
    pass
