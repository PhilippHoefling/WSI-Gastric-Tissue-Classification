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

from pathlib import Path

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



def get_storage_name(targetfolder: str, model_name: str, timestampStr: str):
    '''
    Create folder directory for a model. In this folder the model and all associated files
    return: Directory of the new model
    '''

    folderpath = Path(targetfolder + "/" + model_name + "_model" + "_" + timestampStr)
    folderpath.mkdir(parents=True, exist_ok=True)

    return folderpath
def store_hyperparameters(target_dir_new_model: str, model_name: str, dict: dict, timestampStr: str):
    '''
    Save the hyperparameters of the trained model in the model folder directory
    return: Directory of the new model
    '''

    folderpath = get_storage_name(target_dir_new_model, model_name, timestampStr)

    dict_path = folderpath / ("hyperparameter_dict.pkl")
    with open(dict_path, "wb") as filestore:
        pickle.dump(dict, filestore)
    return folderpath

def store_model(target_dir_new_model: str, tf_model: bool, model_name: str, hyperparameter_dict: dict,
                trained_epochs: int, classifier_model: torch.nn.Module, results: dict, batch_size: int,
                total_train_time: float, timestampStr: str, used_combination: int):
    '''
    Store all files related to the model in the model directory. (Hyperparameters, model summary, figures, and results)
    It also creates or updates a csv-file where all training informations, the model path, and the used hyperparameters are stored
    return: Directory of the new model
    '''

    logger.info("Store model, results and hyperparameters...")

    folderpath = store_hyperparameters(target_dir_new_model, model_name, hyperparameter_dict, timestampStr)
    model_path = folderpath / ("model.pkl")
    results_path = folderpath / ("results.pkl")
    summary_path = folderpath / ("summary.pkl")

    model_summary = summary(model=classifier_model,
                            input_size=(batch_size, 3, 224, 224),  # make sure this is "input_size", not "input_shape"
                            col_names=["input_size", "output_size", "num_params", "trainable"],
                            col_width=20,
                            row_settings=["var_names"],
                            verbose=0
                            )

    with open(summary_path, "wb") as filestore:
        pickle.dump(model_summary, filestore)

    with open(model_path, "wb") as filestore:
        pickle.dump(classifier_model, filestore)

    with open(results_path, "wb") as filestore:
        pickle.dump(results, filestore)

    df = pd.DataFrame()
    df["model_type"] = [model_name]
    df["model_path"] = [folderpath]
    df["pretrained"] = [tf_model]
    if used_combination == -1:
        pass
    else:
        df["used_dataaugmention_combination"] = ["comb_"+str(used_combination)]
    df["epochs"] = [hyperparameter_dict["epochs"]]
    df["seed"] = [hyperparameter_dict["seed"]]
    df["learning_rate"] = [hyperparameter_dict["learning_rate"]]
    df["dropout"] = [hyperparameter_dict["dropout"]]
    df["batch_size"] = [hyperparameter_dict["batch_size"]]
    df["num_workers"] = [hyperparameter_dict["num_workers"]]
    df["total_train_time"] = [total_train_time / 60]
    df["trained_epochs"] = [trained_epochs]
    df["train_loss"] = [list(results["train_loss"])[-1]]
    df["train_acc"] = [list(results["train_acc"])[-1]]
    df["val_loss"] = [list(results["val_loss"])[-1]]
    df["val_acc"] = [list(results["val_acc"])[-1]]

    update_df = False
    path = Path('models/models_results.csv')

    if path.is_file() == True:
        df_exist = pd.read_csv('models/models_results.csv')
        for i in range(df_exist.shape[0]):
            if Path(df["model_path"][0]) == Path(df_exist["model_path"].iloc[i]):
                logger.info("Update model results in csv file")
                df_exist.loc[i, "total_train_time"] = df["total_train_time"][0]
                df_exist.loc[i, "trained_epochs"] = df["trained_epochs"][0]
                df_exist.loc[i, "train_loss"] = df["train_loss"][0]
                df_exist.loc[i, "train_acc"] = df["train_acc"][0]
                df_exist.loc[i, "val_loss"] = df["val_loss"][0]
                df_exist.loc[i, "val_acc"] = df["val_acc"][0]
                update_df = True
            else:
                continue

        if update_df == True:
            df_exist.to_csv('models/models_results.csv', index=False)
        else:
            logger.info("Add new model results in csv file")
            df_new = pd.concat([df_exist, df], ignore_index=True)
            df_new.to_csv('models/models_results.csv', index=False)
    else:
        logger.info("Create csv file for storing model results")
        df.to_csv('models/models_results.csv', index=False)

    logger.info("Model stored!")

    return folderpath

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
    Validation step for the selected model and calculating the validation loss
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

