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

    splits = ['test','train','val']

    # Set the random seeds
    torch.manual_seed(cfg_hp["seed"][s])
    torch.cuda.manual_seed(cfg_hp["seed"][s])

    labels = ['antrum','corpus','intermediate']

    #load and sort all tiles from tileexporter in the according folder

    folders, num_folders = loading(src_dir)
    print(folders)
    for folder in folders:
     subfolders, num_folders = loading(src_dir)
     dir_folder = src_dir + "/" + folder

    # for subfolder in subfolders:
    #     dir_subsubfolder = dir_folder + "/" + subfolder
    #     images, num_images = loading(dir_subsubfolder)
    #     for image in images:
    #          dir_image = dir_subsubfolder + "/" + image
    #         for label in labels:
    #            if label in image:
    #                shutil.copyfile(dir_image, dst_dir + label + "/" + image)

load_sort_data("D:/DigPat2/tiles","C:/Users/phili/DataspellProjects/xAIMasterThesis/data/Processed/")

#%%
#to do Eine rekursive Funktion schreiben, welche man einen Ordner übergibt, die checkt ob in dem Ordner PNGs sind
#Falls nicht in die Unterodner geht