import os
import math
import copy

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from config import config_hyperparameter as cfg_hp
import torch
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
# import skimage
from skimage.io import imread

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#manualtransformation placeholder for augmentation
def load_data(train_dir: str, val_dir: str, num_workers: int, batch_size: int, augmentation: bool,comb_1: bool,comb_2: bool,comb_3:):
    '''
    Load the data into data loaders with the choosen transformation function
    return: dataloaders for training and validation, list of the class names in the dataset
    '''

    # Load transform function with or without data augmentation
    if augmentation:
        manual_transforms = manual_transformation(comb_1=comb_1,comb_2=comb_2,comb_3=comb_3)
    else:
        manual_transforms = transforms.Compose([
            transforms.Resize((384,384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


    # Create train and validation data loaders as well as get a list of class names
    train_dataloader, val_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                       val_dir=val_dir,
                                                                       transform=manual_transforms,
                                                                       batch_size=batch_size,
                                                                       num_workers=num_workers)

    return train_dataloader, val_dataloader, class_names

def create_dataloaders(train_dir: str,
                       val_dir: str,
                       transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int = 1
                       ):
    '''
    Create train and validation data loaders from Image folders
    return: Dataloaders for training and validation, list of the class names in the dataset
    '''

    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True
                                  )

    val_dataloader = DataLoader(val_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True
                                )

    return train_dataloader, val_dataloader, class_names