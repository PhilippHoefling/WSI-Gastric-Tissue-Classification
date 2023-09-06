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



def load_pretrained_model(device, tf_model: str, class_names:list, dropout: int):
    '''
    Load the pretrainind ResNet50 pytorch model with or without weights
    return: model and the pretrained weights
    '''
    # Set the manual seeds
    torch.manual_seed(cfg_hp["seed"])
    torch.cuda.manual_seed(cfg_hp["seed"])
    # Load weights from
    weights = torchvision.models.ResNet18_Weights.DEFAULT

    # Load pretrained model with or without weights
    if tf_model =='imagenet':
        # Load pretrained ResNet50 Model
        model = torchvision.models.resnet18(weights)

    #elif tf_model =='PathDat':
    #    model = resnet50(pretrained=True, progress=False, key="BT")
    #    return model
    else:
        model = torchvision.models.resnet18()




    # Remove the last average pooling layer and the fully connected layer
    #model = nn.Sequential(*list(model.children())[:-1])

    #print(model)
    num_ftrs = model.fc.in_features
    # Recreate classifier layer with an additional layer in between
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=num_ftrs, out_features=1))
    #    nn.ReLU(),
    #    nn.Dropout(dropout),
      #  torch.nn.Linear(in_features=256, out_features=len(class_names))


    print(model)

    # Unfreeze all the layers
    for param in model.parameters():
        param.requires_grad = True

    # Speed up training
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    return model

def train_new_model(dataset_path: str, tf_model: str):
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
    for b in range(len(cfg_hp["batch_size"])):
        for l in range(len(cfg_hp["lr"])):
            for d in range(len(cfg_hp["dropout"])):
                # Load data
                train_dataloader, val_dataloader, class_names = load_data(train_dir=train_dir,
                                                                          val_dir=val_dir,
                                                                          num_workers=cfg_hp["num_workers"],
                                                                          batch_size=cfg_hp["batch_size"][b],
                                                                          )

                # Load pretrained model, weights and the transforms
                model = load_pretrained_model(device, tf_model=tf_model, class_names=class_names, dropout= d)

                # Define loss and optimizer
                loss_fn = nn.BCEWithLogitsLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg_hp["lr"][l], weight_decay=5e-4)

                #MultiStepLR learning rate scheduler
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20 , gamma=0.1)


                # Set the random seeds
                torch.manual_seed(cfg_hp["seed"])
                torch.cuda.manual_seed(cfg_hp["seed"])

                hyperparameter_dict = {"epochs": cfg_hp["epochs"], "seed": cfg_hp["seed"],
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
                                              scheduler=scheduler,
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
          scheduler: torch.optim.lr_scheduler,
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
                                           scheduler=scheduler,
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
                                       model, results, batch_size, total_train_time, timestampStr)

            early_stopping = 0

        if epoch == 40:
            model_folder = store_model(target_dir_new_model, tf_model, model_name, hyperparameter_dict, trained_epochs,
                                       model, results, batch_size, total_train_time, timestampStr)
            break

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
               scheduler: torch.optim.lr_scheduler,
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

        # Add an extra dimension to the target tensor
        y = y.unsqueeze(1).float()

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
        y_pred_class = torch.sigmoid(y_pred).round()
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    scheduler.step()
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc
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

def get_storage_name(targetfolder: str, model_name: str, timestampStr: str):
    '''
    Create folder directory for a model. In this folder the model and all associated files
    return: Directory of the new model
    '''

    folderpath = Path(targetfolder + "/" + model_name + "_model" + "_" + timestampStr)
    folderpath.mkdir(parents=True, exist_ok=True)

    return folderpath

def store_model(target_dir_new_model: str, tf_model: bool, model_name: str, hyperparameter_dict: dict,
                trained_epochs: int, classifier_model: torch.nn.Module, results: dict, batch_size: int,
                total_train_time: float, timestampStr: str):
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


    with torch.no_grad():
        # Loop through DataLoader batches
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Add an extra dimension to the target tensor
            y = y.unsqueeze(1).float()

            # 1. Forward pass
            val_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            # Calculate and accumulate accuracy
            val_pred_labels =  torch.sigmoid(val_pred_logits).round()
            val_acc += ((val_pred_labels == y).sum().item() / len(val_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc

def plot_loss_acc_curves(model_folder: str):
    # Plots training curves of a results dictionary and saves figures into model directory
    #return: Nothing
    trained_model, model_results, dict_hyperparameters, summary = get_model(Path(model_folder))
    loss = model_results["train_loss"]
    val_loss = model_results["val_loss"]

    accuracy = model_results["train_acc"]
    val_accuracy = model_results["val_acc"]

    epochs = range(len(model_results["train_loss"]))
    print(model_results)

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    axl = plt.gca()
    axl.set_ylim([0, 1.4])
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    axa = plt.gca()
    axa.set_ylim([0.3, 1])
    plt.savefig(model_folder + "/" + "train_loss_acc.png")
    plt.show()

#%%
