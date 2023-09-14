
from timm.models.vision_transformer import VisionTransformer
import os
import math
import copy
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
from config import config_hyperparameter as cfg_hp
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from loguru import logger
from timeit import default_timer as timer
from datetime import datetime
from tqdm.auto import tqdm
from torchinfo import summary
import time
import pickle
import random
# import skimage
from skimage.io import imread
from typing import Dict, List, Tuple
#from sklearn.model_selection import train
import torchvision
from torch import nn
from sklearn.metrics import accuracy_score

from torchvision.models.resnet import Bottleneck, ResNet
from evaluation import get_model

from pathlib import Path
from auxiliaries import store_hyperparameters, store_model, plot_loss_acc_curves
from BaseLine_AntrumCorpus import train

def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url

class VisionTransformerWithCustomHead(nn.Module):
    def __init__(self, model, num_classes):
        super(VisionTransformerWithCustomHead, self).__init__()
        self.model = model  # The ViT model without the head
        self.head = nn.Linear(384, num_classes)  # Modify the output size accordingly

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x

def vit_small(pretrained, progress, key, num_classes, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model = VisionTransformer(
        img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(
            torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        )
        print(verbose)
    # Create an instance of VisionTransformerWithCustomHead
    custom_model = VisionTransformerWithCustomHead(model, num_classes)
    return custom_model


def trainVIT(dataset_path: str):
    model = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16, num_classes=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn= nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_hp["lr"][0])
    #MultiStepLR learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20 , gamma=0.1)

    train_dir = dataset_path + "/train"
    val_dir = dataset_path + "/val"


    hyperparameter_dict = {"epochs": cfg_hp["epochs"], "seed": cfg_hp["seed"],
                           "learning_rate": cfg_hp["lr"], "dropout": cfg_hp["dropout"],
                           "batch_size": cfg_hp["batch_size"], "num_workers": cfg_hp["num_workers"]}

    train_dataloader, val_dataloader, class_names = load_data(train_dir=train_dir,
                                                              val_dir=val_dir,
                                                              num_workers=cfg_hp["num_workers"],
                                                              batch_size=cfg_hp["batch_size"][0],
                                                              )
    model_name= "VisionTransformer"
    target_dir_new_model = 'models'

    results, model_folder= train(target_dir_new_model=target_dir_new_model,
                                  tf_model=True,
                                  model_name=model_name,
                                  model=model,
                                  train_dataloader=train_dataloader,
                                  val_dataloader=val_dataloader,
                                  optimizer=optimizer,
                                  scheduler=scheduler,
                                  loss_fn=loss_fn,
                                  batch_size=cfg_hp["batch_size"],
                                  epochs=cfg_hp["epochs"],
                                  hyperparameter_dict=hyperparameter_dict,
                                  device=device
                                  )



def create_dataloaders(train_dir: str,
                       val_dir: str,
                       val_transform: transforms.Compose,
                       train_transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int = 4
                       ):
    '''
    Create train and validation data loaders from Image folders
    return: Dataloaders for training and validation, list of the class names in the dataset
    '''

    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=val_transform)

    # Get class names
    class_names = train_data.classes
    print(class_names)
    # Turn images into data loaders
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True
                                  )

    val_dataloader = DataLoader(val_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True
                                )

    return train_dataloader, val_dataloader, class_names

#manualtransformation placeholder for augmentation
def load_data(train_dir: str, val_dir: str, num_workers: int, batch_size: int):
    '''
    Load the data into data loaders with the choosen transformation function
    return: dataloaders for training and validation, list of the class names in the dataset
    '''


    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=180),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Create train and validation data loaders as well as get a list of class names
    train_dataloader, val_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                       val_dir=val_dir,
                                                                       train_transform=train_transforms,
                                                                       val_transform=val_transforms,
                                                                       batch_size=batch_size,
                                                                       num_workers=num_workers)
    print(class_names)
    return train_dataloader, val_dataloader, class_names



def loading(folder_name: str):
    direc = os.path.join(
        folder_name)
    return os.listdir(direc), len(os.listdir(direc))


#%%
