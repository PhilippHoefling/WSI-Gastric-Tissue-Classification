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
import splitfolders
import random
# import skimage
from skimage.io import imread

from sklearn.model_selection import train
import torchvision
from torch import nn_test_split
from sklearn.metrics import accuracy_score

from torchvision.models.resnet import Bottleneck, ResNet

from pathlib import Path

def create_dataloaders(train_dir: str,
                       val_dir: str,
                       transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int = 4
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

#manualtransformation placeholder for augmentation
def load_data(train_dir: str, val_dir: str, num_workers: int, batch_size: int):
    '''
    Load the data into data loaders with the choosen transformation function
    return: dataloaders for training and validation, list of the class names in the dataset
    '''

    # Load transform function with or without data augmentation
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



def loading(folder_name: str):
    direc = os.path.join(
        folder_name)
    return os.listdir(direc), len(os.listdir(direc))
def split(original_dataset_dir: str,seed: int):
    #Truncate the existing combined dataset folder
    dst_dir = 'data_combined'
    splits, num_splits = loading(dst_dir)
    for split in splits:
        if split == 'train' or split == 'val':
            split_dir = dst_dir + '/' + split
            shutil.rmtree(split_dir)

    #Split original datasets into train&validation (80/20) and store it as combined dataset
    datasets, num_datasets = loading(original_dataset_dir)
    for dataset in datasets:
        src = original_dataset_dir + '/' + dataset
        dst = 'data_combined'
        splitfolders.ratio(src, output=dst, seed=seed, ratio=(0.8, 0.2))

    rgba_to_rgb()

def train_new_model(dataset_path: str, tf_model: bool, activate_augmentation: bool):
    '''
    Initializes the directories of the dataset, stores the selected model type, chooses the availabe device, initialize the model,
    loads the data, adjustes the last layer of the model architecture, Initializes the loss and the optimizer, sets seeds.
    creates a dictionary to store the used hyperparameters, grid search hyperparameter tuning, excecutes the training process
    return: The directory of the new trained model
    '''

    train_dir = dataset_path + "/train"
    val_dir = dataset_path + "/val"
    target_dir_new_model = 'models'
    if tf_model:
        model_name = "TransferLearning"
    else:
        model_name = "Baseline"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on:")
    logger.info(device)

    for s in range(len(cfg_hp["seed"])):
        split_dataset = False
        for b in range(len(cfg_hp["batch_size"])):
            for l in range(len(cfg_hp["lr"])):
                for d in range(len(cfg_hp["dropout"])):
                    if split_dataset:
                        split(original_dataset_dir='data_original', seed=cfg_hp["seed"][s])
                        split_dataset = False

                    # Load pretrained model, weights and the transforms
                    model, weights = load_pretrained_model(device, tf_model=tf_model)

                    # Load data
                    train_dataloader, val_dataloader, class_names = load_data(train_dir=train_dir,
                                                                              val_dir=val_dir,
                                                                              num_workers=cfg_hp["num_workers"],
                                                                              batch_size=cfg_hp["batch_size"][b],
                                                                              )

                    # Recreate classifier layer
                    model = recreate_classifier_layer(model=model,
                                                      tf_model=tf_model,
                                                      dropout=cfg_hp["dropout"][d],
                                                      class_names=class_names,
                                                      seed=cfg_hp["seed"][s],
                                                      device=device
                                                      )

                    # Define loss and optimizer
                    loss_fn = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_hp["lr"][l])

                    # Set the random seeds
                    torch.manual_seed(cfg_hp["seed"][s])
                    torch.cuda.manual_seed(cfg_hp["seed"][s])

                    hyperparameter_dict = {"epochs": cfg_hp["epochs"], "seed": cfg_hp["seed"][s],
                                           "learning_rate": cfg_hp["lr"][l], "dropout": cfg_hp["dropout"][d],
                                           "batch_size": cfg_hp["batch_size"][b], "num_workers": cfg_hp["num_workers"]}


                    # Setup training and save the results
                    results, model_folder = train(target_dir_new_model=target_dir_new_model,
                                                  tf_model=tf_model,
                                                  model_name=model_name,
                                                  model=model,
                                                  train_dataloader=train_dataloader,
                                                  val_dataloader=val_dataloader,
                                                  optimizer=optimizer,
                                                  loss_fn=loss_fn,
                                                  batch_size=cfg_hp["batch_size"][b],
                                                  epochs=cfg_hp["epochs"],
                                                  hyperparameter_dict=hyperparameter_dict,
                                                  device=device
                                                  )

    return model_folder

def train(target_dir_new_model: str,
          tf_model: bool,
          model_name: str,
          model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          batch_size: int,
          epochs: int,
          hyperparameter_dict: dict,
          device
          ) -> Dict[str, List]:
    '''
    Iteration over each epoch for training and validation of the select model. It also calls early stopping if the model
    does not improve over [patience] number of epochs.
    return: Results and the directory of the new trained model
    '''

    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []
               }

    # Start the timer
    start_time = timer()

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d%m%Y_%H%M")

    # Auxilary variables
    early_stopping = 0
    max_acc = 0
    trained_epochs = 0
    model_folder = ''

    # Loop through training and valing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        trained_epochs = epoch + 1
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device
                                           )
        val_loss, val_acc = val_step(model=model,
                                     dataloader=val_dataloader,
                                     loss_fn=loss_fn,
                                     device=device
                                     )

        # Print out what's happening
        print(
            f"\nEpoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        # Early Stopping
        max_acc = max(results["val_acc"])
        if results["val_acc"][-1] < max_acc:
            early_stopping = early_stopping + 1
        else:
            # End the timer and print out how long it took
            end_time = timer()

            time.sleep(10)
            total_train_time = end_time - start_time
            model_folder = store_model(target_dir_new_model, tf_model, model_name, hyperparameter_dict, trained_epochs,
                                       model, results, batch_size, total_train_time, timestampStr,used_combination)

            early_stopping = 0

        if epoch < 9:
            early_stopping = 0

        if early_stopping == cfg_hp["patience"]:
            break
        else:
            continue

    return results, model_folder

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device
               ) -> Tuple[float, float]:
    '''
    Train step for the selected model (Baseline or Transfer Learning model) and calculating the train loss
    return: Train loss
    '''

    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device
             ) -> Tuple[float, float]:
    '''
    Validation step for the selected model (Baseline or Transfer Learning model) and calculating the validation loss
    return: Validation loss
    '''
    # Put model in eval mode
    model.eval()

    # Setup val loss and val accuracy values
    val_loss, val_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            val_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            # Calculate and accumulate accuracy
            val_pred_labels = val_pred_logits.argmax(dim=1)
            val_acc += ((val_pred_labels == y).sum().item() / len(val_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc

