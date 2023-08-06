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
import glob



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
              #check if the image contains more than 5% tissue
              for label in labels:
                 if label in image and is_white_or_grey_png(dir_image):
                     if folder in splits["train"]:
                         shutil.copyfile(dir_image, dst_dir+ "/train/" + label + "/" + image)
                     if folder in splits["validate"]:
                         shutil.copyfile(dir_image, dst_dir+ "/val/" + label + "/" + image)
                     if folder in splits["test"]:
                         shutil.copyfile(dir_image, dst_dir+ "/test/" + label + "/" + image)

def count_png_files_in_subfolders(folder_path, subfolder_names):
    png_counts = []
    for subfolder_name in subfolder_names:
        subfolder_path = os.path.join(folder_path, subfolder_name)
        png_count = sum(1 for file in glob.glob(os.path.join(subfolder_path, "*.png")))
        png_counts.append(png_count)
    return png_counts

def plot_file_distribution(dataset_path):
    folders, num_folders = loading(dataset_path)

    png_counts = dict()

    for folder in folders:
        print(folder)
        dir_folder = dataset_path + "/" + folder
        subfolders, num_folders = loading(dir_folder)
        png_counts[folder] = count_png_files_in_subfolders(dir_folder, subfolders)


    print(png_counts)

    # Plot the data distribution using a bar chart
    x = np.arange(len(cfg_hp["class_names"]))
    width = 0.2

    plt.bar(x, png_counts[folders[0]], width, label=folders[0])
    plt.bar([pos + width for pos in x], png_counts[folders[1]], width, label=folders[1])
    plt.bar([pos + 2 * width for pos in x], png_counts[folders[2]], width, label=folders[2])

    plt.xlabel('Class Labels')
    plt.ylabel('Number of Samples')
    plt.title('Data Distribution of Train, Validation, and Test Datasets')
    plt.xticks([pos + width for pos in x], cfg_hp["class_names"])
    plt.legend()
    plt.show()
def is_white_or_grey_png(image_path, threshold=0.95):
    try:
        image = Image.open(image_path)
    except IOError:
        raise ValueError("Unable to open the image file")

    # Convert the image to grayscale for easy white/grey detection
    grayscale_image = image.convert("L")

    # Get the pixel data from the image
    pixels = grayscale_image.load()

    # Get the image size
    width, height = image.size

    # Count the number of white/grey pixels in the image
    white_or_grey_pixel_count = sum(pixels[x, y] >= 200 for x in range(width) for y in range(height))

    # Calculate the percentage of white/grey pixels in the image
    white_or_grey_percentage = white_or_grey_pixel_count / float(width * height)

    # Check if the white/grey percentage is above the threshold
    return white_or_grey_percentage < threshold

# Usage example



#%%
#to do Eine rekursive Funktion schreiben, welche man einen Ordner übergibt, die checkt ob in dem Ordner PNGs sind
#Falls nicht in die Unterodner geht