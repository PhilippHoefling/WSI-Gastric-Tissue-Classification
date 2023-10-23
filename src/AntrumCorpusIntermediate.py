import os

import matplotlib.pyplot as plt
from config import config_hyperparameter as cfg_hp
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from loguru import logger
from timeit import default_timer as timer
from datetime import datetime
from tqdm.auto import tqdm
import time
# import skimage
from typing import Dict, List, Tuple
#from sklearn.model_selection import train
import torchvision
from torch import nn
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, f1_score, recall_score

from pathlib import Path
from Tile_inference import get_model
from auxiliaries import  store_model

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
        # Load pretrained ResNet18 Model
        model = torchvision.models.resnet18(weights)

    #elif tf_model =='PathDat':
    #    model = resnet50(pretrained=True, progress=False, key="BT")
    #    return model
    else:
        model = torchvision.models.resnet18()

    num_ftrs = model.fc.in_features
    # Recreate classifier layer with an additional layer in between
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=num_ftrs, out_features=3))


    # Unfreeze all the layers
    for param in model.parameters():
        param.requires_grad = True

    # Speed up training
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    return model

def train_new_3model(dataset_path: str, tf_model: str):
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
                loss_fn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg_hp["lr"][l], weight_decay=1e-4)

                #learning rate scheduler
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

def print_3model_metrices(model_folder, test_folder):
    trained_model, model_results, dict_hyperparameters, summary = get_model(Path(model_folder))
    image_path_list = list(Path(test_folder).glob("*/*.*"))
    class_names = cfg_hp["class_names"]
    accuracy = []
    predictions = []
    y_test = []

    for image_path in image_path_list:
        # Load in image and convert the tensor values to float32
        target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

        # Divide the image pixel values by 255 to get them between [0, 1]
        target_image = target_image / 255


        manual_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Transform if necessary
        target_image = manual_transforms(target_image)

        # Turn on model evaluation mode and inference mode
        trained_model.eval()
        with torch.inference_mode():
            # Add an extra dimension to the image
            target_image = target_image.unsqueeze(dim=0)

            # Make a prediction on image with an extra dimension
            target_image_pred = trained_model(target_image.cuda())

        # Convert logits -> prediction probabilities
        # (using torch.softmax() for multi-class classification)
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

        # Convert prediction probabilities -> prediction labels
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
        pred_class = class_names[target_image_pred_label.item()]
        true_class = image_path.parts[3]
        predictions.append(target_image_pred_label.item())
        y_test.append(class_names.index(true_class))
        if pred_class == true_class:
            accuracy.append(1)
        else:
            accuracy.append(0)

    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=class_names, cmap='Blues',
                                            colorbar=False)
    plt.savefig(model_folder + '/test_confusion_matrix.png')
    plt.show()

    print("Accuracy on test set: " + str(sum(accuracy) / len(accuracy) * 100) + " %")
    print("Precision on test set " + str(precision_score(y_test, predictions, average='macro')))
    print("Recall on test set " + str(recall_score(y_test, predictions, average='macro')))
    print("F1 Score on test set " + str(f1_score(y_test, predictions, average='macro')))
    # print("Log-Loss on test set " + str(log_loss(y_test, predictions)))


#%%