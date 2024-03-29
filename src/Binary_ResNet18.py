
from config import config_hyperparameter as cfg_hp
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from loguru import logger
from timeit import default_timer as timer
from datetime import datetime
from tqdm.auto import tqdm
import time

from typing import Dict, List, Tuple

import torchvision
from torch import nn
from auxiliaries import store_model

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


    # Base transforms
    train_transforms =  transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        transforms.RandomRotation(degrees=180),
        transforms.Resize((224, 224)),
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

def load_pretrained_model(device, tf_model: str):
    '''
    Load the pretrainind ResNet18 pytorch model with or without weights
    return: model and the pretrained weights
    '''
    # Set the manual seeds
    torch.manual_seed(cfg_hp["seed"])
    torch.cuda.manual_seed(cfg_hp["seed"])

    # Load pretrained model with or without weights
    if tf_model =='imagenet':
        # Load pretrained ResNet18 Model
        model = torchvision.models.resnet18(pretrained=True, progress=False)
    else:
        model = torchvision.models.resnet18(pretrained=False, progress= False)

    num_ftrs = model.fc.in_features
    # Recreate classifier layer with dropout and a linear layer
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=num_ftrs, out_features=1)
    )


    # Unfreeze all the layers
    for param in model.parameters():
        param.requires_grad = True


    # Speed up training
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    return model

def train_new_binary_model(dataset_path: str, tf_model: str):
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
    #Augmentation to test


    for b in cfg_hp["batch_size"]:
        for l in cfg_hp["lr"]:

            print(l)
            # Load data
            train_dataloader, val_dataloader, class_names = load_data(train_dir=train_dir,
                                                                      val_dir=val_dir,
                                                                      num_workers=cfg_hp["num_workers"],
                                                                      batch_size=b
                                                                      )

            # Load pretrained model, weights and the transforms
            model = load_pretrained_model(device, tf_model=tf_model)

            # Define loss and optimizer
            loss_fn = nn.BCEWithLogitsLoss()
            #optimizer = torch.optim.Adam(model.parameters(), l)

            #learning rate scheduler

            #scheduler = None

            # SGD Optimizer
            optimizer = torch.optim.SGD(model.parameters(), lr=l, momentum=0.9, weight_decay=0.0001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=75, eta_min=0)
            # Define the learning rate scheduler
            #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

            # Set the random seeds
            torch.manual_seed(cfg_hp["seed"])
            torch.cuda.manual_seed(cfg_hp["seed"])

            hyperparameter_dict = {"epochs": cfg_hp["epochs"], "seed": cfg_hp["seed"],
                                   "learning_rate": l, "dropout": cfg_hp["dropout"],
                                   "batch_size":b, "num_workers": cfg_hp["num_workers"]}


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
                                          batch_size=b,
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
               "train_sens": [],
               "train_spez": [],
               "val_loss": [],
               "val_acc": [],
               "val_sens": [],
               "val_spez": [],
               }

    # Start the timer
    start_time = timer()

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d%m%Y_%H%M")

    # Auxilary variables
    early_stopping = 0
    min_val_loss = float('inf')
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
                                           device=device,
                                           trained_epochs = trained_epochs
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

        min_val_loss = min(results["val_loss"])
        # Early Stopping
        if results["val_loss"][-1] > min_val_loss:
           early_stopping = early_stopping + 1
        else:
           # End the timer and print out how long it took
            end_time = timer()

            time.sleep(10)
            total_train_time = end_time - start_time
            model_folder = store_model(target_dir_new_model, tf_model, model_name, hyperparameter_dict, trained_epochs,
                                       model, results, batch_size, total_train_time, timestampStr)
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
               device: torch.device,
               trained_epochs: int
               ) -> Tuple[float, float]:
    '''
    Train step for the selected model (Baseline or Transfer Learning model) and calculating the train loss
    and accuracy
    return: Train loss and Train accuracy
    '''

    # Set model to training mode
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0


    for data, labels in dataloader:
        # Move data and labels to the specified device (e.g., GPU)
        data, labels = data.to(device), labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Add an extra dimension to the target tensor
        labels = labels.unsqueeze(1).float()

        # Forward pass: compute predictions
        outputs = model(data)

        # Compute loss between the predicted outputs and labels
        loss = loss_fn(outputs, labels)

        # Compute gradients
        loss.backward()

        # Update the model parameters
        optimizer.step()

        total_loss += loss.item()

        # Predict class labels and count the number of correct predictions
        predicted = torch.sigmoid(outputs).round()

        #Update Accuracy
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    # Adjust the learning rate based on the scheduler
    #if trained_epochs > 10:
    scheduler.step()


    # Calculate average loss, accuracy, sensitivity, and specificity
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
    and validation accuracy
    return: Validation loss and validation accuracy
    '''
    # Set the model to evaluation mode (affects dropout and batch normalization)
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0


    # Disable gradient computation during validation
    with torch.no_grad():
        for data, labels in dataloader:
            # Move data and labels to the specified device (e.g., GPU)
            data, labels = data.to(device), labels.to(device)

            # Forward pass: compute predictions
            outputs = model(data)

            # Compute loss between the predicted outputs and labels
            loss = loss_fn(outputs, labels.unsqueeze(1).float())
            total_loss += loss.item()

            # Predict class labels and count the number of correct predictions
            predicted = torch.sigmoid(outputs).round()
            correct_predictions += (predicted == labels.unsqueeze(1)).sum().item()
            total_samples += labels.size(0)


        # Calculate average loss, accuracy, sensitivity, and specificity
        average_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples

    return average_loss, accuracy
