
import timm
import os
from src.config import config_hyperparameter as cfg_hp
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from datetime import datetime
from tqdm.auto import tqdm
import time
# import skimage
from typing import Dict, List, Tuple
#from sklearn.model_selection import train
from torch import nn

from src.auxiliaries import store_model


def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url



def vit_small(pretrained, progress, key, num_classes, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    model =timm.create_model("vit_base_patch16_224", pretrained=True)
    #VisionTransformer(
    #    img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=1, pretrained=True )
    #if pretrained:
        #pretrained_url = get_pretrained_url(key)
        #verbose = model.load_state_dict(
        #    torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        #)
        #print(verbose)
    # Modify Vision Transformer Head to Binary Output
    model.head = nn.Linear(model.head.in_features, 1)
    return model


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

    return model_folder


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
               device: torch.device
               ) -> Tuple[float, float]:
    '''
    Train step for the selected model (Baseline or Transfer Learning model) and calculating the train loss
    return: Train loss
    '''

    model.train()  # Set model to training mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()  # Reset gradients

        # Add an extra dimension to the target tensor
        labels = labels.float().unsqueeze(1)

        outputs = model(data)  # Forward pass
        loss = loss_fn(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

        # Update the learning rate
        #scheduler.step()

        total_loss += loss.item()

        predicted = torch.round(torch.sigmoid(outputs))
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)



    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return average_loss, accuracy


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device
             ) -> Tuple[float, float]:
    '''
    Validation step for the selected model (Baseline or Transfer Learning model) and calculating the validation loss
    return: Validation loss
    '''
    model.eval()  # set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # disable gradient computation during validation
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)

            labels = labels.float().unsqueeze(1)
            outputs = model(data)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))

            correct_predictions += (predicted == labels.unsqueeze(1)).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy

#%%
