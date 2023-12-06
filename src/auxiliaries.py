import os

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from loguru import logger
from torchinfo import summary
import pickle
# import skimage
#from sklearn.model_selection import train


from pathlib import Path




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


def plot_loss_acc_curves(model_folder: str):
    # Assuming get_model is a function you've defined elsewhere that loads your model and results
    trained_model, model_results, dict_hyperparameters, summary = get_model(Path(model_folder))

    loss = model_results["train_loss"]
    val_loss = model_results["val_loss"]
    accuracy = model_results["train_acc"]
    val_accuracy = model_results["val_acc"]
    epochs = range(len(loss))

    # Set the global font size
    plt.rcParams.update({'font.size': 14})

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    axl = plt.gca()
    axl.set_ylim([0, 1.4])
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    axa = plt.gca()
    axa.set_ylim([0.3, 1])
    plt.legend()

    plt.savefig(Path(model_folder) / "train_loss_acc.png")
    plt.show()

    plot_spec_sens_curves(model_folder = model_folder)

def plot_spec_sens_curves(model_folder: str):
    # Assuming get_model is a function you've defined elsewhere that loads your model and results
    trained_model, model_results, dict_hyperparameters, summary = get_model(Path(model_folder))

    sens = model_results["train_sens"]
    val_sens = model_results["val_sens"]
    spez = model_results["train_spez"]
    val_spez = model_results["val_spez"]
    epochs = range(len(sens))

    # Set the global font size
    plt.rcParams.update({'font.size': 14})

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, sens, label="Training Sensitivity")
    plt.plot(epochs, val_sens, label="Validation Sensitivity")
    plt.title("Sensitivity")
    plt.xlabel("Epochs")
    plt.ylabel("Sensitivity")
    axl = plt.gca()
    axl.set_ylim([0, 1.4])
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, spez, label="Training Specificity")
    plt.plot(epochs, val_spez, label="Validation Specificity")
    plt.title("Specificity")
    plt.xlabel("Epochs")
    plt.ylabel("Specificity")
    axa = plt.gca()
    axa.set_ylim([0.3, 1])
    plt.legend()

    plt.savefig(Path(model_folder) / "Train_Val_Sens_Spez.png")
    plt.show()

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

    print(batch_size)

    if isinstance(batch_size, list):
        batch_size = batch_size[0]
        print(batch_size)

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

def image_entropy(img):
    """Calculate the entropy of an image"""
    # Load the image in color (adjust path accordingly)

    image_np = np.array(img)

    # Convert RGB to BGR
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)


#img = cv2.imread(img_path)

    channels = cv2.split(image_bgr)
    entropy = []

    for ch in channels:
        hist = cv2.calcHist([ch], [0], None, [256], [0,256])
        hist /= hist.sum()
        # Calculate entropy for this channel
        ch_entropy = -np.sum(hist*np.log2(hist + np.finfo(float).eps))
        entropy.append(ch_entropy)


    # Average the entropies of the channels
    entropy /= 3.0

    return entropy[2]

def get_model(model_folder: str):
    # get model from model folder
    model_folder = Path(model_folder)


    # Use the / operator to join paths
    hyperparameters_path = model_folder / "hyperparameter_dict.pkl"
    model_path = model_folder / "model.pkl"
    results_path = model_folder / "results.pkl"
    summary_path = model_folder / "summary.pkl"

    with open(model_path, "rb") as fid:
        classifier_model = pickle.load(fid)

    with open(results_path, "rb") as fid:
        results = pickle.load(fid)

    with open(hyperparameters_path, "rb") as fid:
        dict = pickle.load(fid)

    with open(summary_path, "rb") as fid:
        summary = pickle.load(fid)

    logger.info("Model and hyperparameters loaded!")
    logger.info("The model is trained with the following hyperparameters:")
    logger.info(dict)

    return classifier_model, results, dict, summary
