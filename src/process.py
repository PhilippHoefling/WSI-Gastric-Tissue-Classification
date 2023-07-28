#Data Preprocessing

# ## Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pandas as pd
import os
import torch
from config import config_hyperparameter as cfg_hp
import shutil
import splitfolders



def loading(folder_name: str):
    direc = os.path.join(
        folder_name)
    return os.listdir(direc), len(os.listdir(direc))

def load_sort_data(src_dir, dst_dir):
    #check if src dir exists
    if not os.path.exists(src_dir) or not os.path.isdir(src_dir):
            print(f"Source path '{src_dir}' is not a valid directory.")

    # Set the random seed
    np.random.seed(cfg_hp["seed"])

    labels = ['antrum','corpus','intermediate']


    folders, num_folders = loading(src_dir)
    npfol = np.array(folders)
    np.random.shuffle(npfol)

    train, validate, test = np.split(npfol, [int(.8*len(npfol)), int(.9*len(npfol))])

    #split dictionary
    splits = {
        "train" : train,
        "validate" : validate,
        "test" : test
    }

    #load and sort all tiles from tileexporter in the according folder
    for folder in folders:
     dir_folder = src_dir + "/" + folder
     subfolders, num_folders = loading(dir_folder)
     for subfolder in subfolders:
         dir_subsubfolder = dir_folder + "/" + subfolder
         images, num_images = loading(dir_subsubfolder)
         for image in images:
              dir_image = dir_subsubfolder + "/" + image
              for label in labels:
                 if label in image:
                     if folder in splits["train"]:
                         shutil.copyfile(dir_image, dst_dir+ "/train/" + label + "/" + image)
                     if folder in splits["validate"]:
                         shutil.copyfile(dir_image, dst_dir+ "/val/" + label + "/" + image)
                     if folder in splits["test"]:
                         shutil.copyfile(dir_image, dst_dir+ "/test/" + label + "/" + image)


#%%
#to do Eine rekursive Funktion schreiben, welche man einen Ordner übergibt, die checkt ob in dem Ordner PNGs sind
#Falls nicht in die Unterodner geht