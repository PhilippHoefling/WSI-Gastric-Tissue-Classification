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
import csv



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

    #labels = 'corpus','antrum', 'intermediate'
    labels = '_inflamed','_noninflamed'

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

    # Open the CSV file in write mode
    with open("data_split.csv", 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the string as a single-row CSV entry
        csv_writer.writerow(["train: " + str(train)])
        csv_writer.writerow(["val: " + str(validate)])
        csv_writer.writerow(["test: " + str(test)])


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
                             shutil.copyfile(dir_image, dst_dir+ "/train/" + label.replace("_","") + "/" + image)
                         if folder in splits["validate"]:
                             shutil.copyfile(dir_image, dst_dir+ "/val/" + label.replace("_","") + "/" + image)
                         if folder in splits["test"]:
                             shutil.copyfile(dir_image, dst_dir+ "/test/" + label.replace("_","") + "/" + image)



def plot_file_distribution(dataset_path):
    folders, num_folders = loading(dataset_path)

    classes = cfg_hp["class_names"]
    datasets = ["train", "val", "test"]

    class_counts = {dataset: {cls: 0 for cls in classes} for dataset in datasets}

    for dataset in datasets:
        for cls in classes:
            class_folder = os.path.join(dataset_path, dataset, cls)
            class_counts[dataset][cls] = len(os.listdir(class_folder))

    num_datasets = len(datasets)
    bar_width = 0.35
    index = np.arange(num_datasets)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, cls in enumerate(classes):
        counts = [class_counts[dataset][cls] for dataset in datasets]
        ax.bar(index + i * bar_width, counts, bar_width, label=cls)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Class Distribution')
    ax.set_xticks(index + bar_width * (num_datasets - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend()

    plt.tight_layout()
    plt.show()






def is_white_or_grey_png(image_path, threshold=0.90):
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





#%%
#to do Eine rekursive Funktion schreiben, welche man einen Ordner übergibt, die checkt ob in dem Ordner PNGs sind
#Falls nicht in die Unterodner geht