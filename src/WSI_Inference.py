from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torchvision import transforms

from config import config_hyperparameter as cfg_hp
import torch.nn as nn
import torch.optim as optim
from auxiliaries import get_model
import math
from Tile_inference import plot_prob_distribution
import csv
import ast

# workaround for Openslide import
OPENSLIDE_PATH = r'C:\Users\phili\OpenSlide\openslide-win64-20230414\bin'
import os

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
from openslide import deepzoom


def WSI_Test_Pipeline(model_folder: str, slidepath: str):
    # load classes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # get model and parameters
    trained_model, model_results, dict_hyperparameters, summary = get_model(model_folder)

    manual_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    trained_model.to(device)
    # Turn on model evaluation mode and inference mode
    trained_model.eval()

    # open slide with openslide
    slide = openslide.open_slide(slidepath)
    # generate tiles with DeepZoomGenerator
    tiles = deepzoom.DeepZoomGenerator(slide, tile_size=224, overlap=112,limit_bounds=False)

    col, rows = tiles.level_tiles[15]
    predictions = np.empty([rows, col])

    for c in range(col):
        for r in range(rows):
            single_tile = tiles.get_tile(15, (c, r))

            single_tile.save('DeepZoomTest/'+str(c) + " " + str(r) + ".png", "PNG")

            # Sample usage
            np_tile = np.array(single_tile)

            # Pink lower bound in HSV
            lower_bound = np.array([135, 40, 40])
            # Purple upper bound in HSV
            upper_bound = np.array([172, 255, 255])

            if is_tile_of_interest(np_tile, lower_bound, upper_bound):
                with torch.inference_mode():
                    # Add an extra dimension to the image
                    single_tile = manual_transforms(single_tile).unsqueeze(dim=0)

                    # Make a prediction on image with an extra dimension
                    target_image_pred = trained_model(single_tile.cuda())
                #apply Sigmoid on Proabbility
                target_image_pred_probs = torch.sigmoid(target_image_pred)
                predictions[r][c] = target_image_pred_probs.tolist()[0][0]
            else:
                predictions[r][c] = -1

    # Visualize Results in a Heatmap
    SlideHeatmap(heatmap_data=predictions)

def TestOnWSISlideFolder(model_folder_inf, model_folder_tissue, testfolder):
    # Path to groundtrouth CSV file
    csv_file_path = 'WSI_Path_Inflamed.csv'
    # Initialize an empty dictionary to store your data
    SlideDict = {}

    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile)

        # Iterate over rows in the CSV
        for row in csv_reader:
            # Assuming the format "test: {'15BHE': ['inflamed'], '15CHE': ['inflamed'], ..."
            # Split the row on ':' to separate the key and the dictionary
            key, dict_str = row[0].split(':', 1)

            # Strip any leading/trailing whitespace and remove the quotation marks
            key = key.strip().replace('"', '')

            # Use ast.literal_eval to safely evaluate the dictionary string
            dict_str = dict_str.strip()
            value_dict = ast.literal_eval(dict_str)

            # Add the key-value pair to your data dictionary
            SlideDict[key] = value_dict

    print(SlideDict)

    class_names = cfg_hp["class_names"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # get model and parameters
    trained_inf_model, model_results, dict_hyperparameters, summary = get_model(model_folder_inf)


    manual_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    trained_inf_model.to(device)

    # Turn on model evaluation mode and inference mode
    trained_inf_model.eval()


    #array for predictions
    inflammation_ratios = []
    inflammation_ratios_withWSI = []
    #list for path building
    infNoninf = {'inflamed': 'inflamed',
                 'noninflamed': 'non-inflamed'}

    for slidename in SlideDict['test']:
        # open slide with openslide

        path = "/mnt/thempel/scans/" + str(infNoninf[SlideDict['test'][slidename][0]]) + "/" +str(slidename) + ".mrxs"
        slide = openslide.open_slide(path)

        tiles = deepzoom.DeepZoomGenerator(slide, tile_size=224,  limit_bounds=True)

        col, rows = tiles.level_tiles[14]
        predictions_inf = np.empty([rows, col])
        rounded_predictions_inf = np.empty([rows, col])


        for c in range(col):
            for r in range(rows):
                single_tile = tiles.get_tile(14, (c, r))

                # Sample usage
                np_tile = np.array(single_tile)

                # Pink lower bound in HSV
                lower_bound = np.array([135, 40, 40])
                # Purple upper bound in HSV
                upper_bound = np.array([172, 255, 255])

                if is_tile_of_interest(np_tile, lower_bound, upper_bound):
                    with torch.inference_mode():
                        # Add an extra dimension to the image
                        single_tile = manual_transforms(single_tile).unsqueeze(dim=0)

                        # Divide the image pixel values by 255 to get them between [0, 1]
                        # target_image = single_tile / 255


                        # Make a prediction on image with an extra dimension
                        target_image_pred_inf = trained_inf_model(single_tile.cuda())


                        target_image_pred_inf_probs = torch.sigmoid(target_image_pred_inf)



                    predictions_inf[r][c] = target_image_pred_inf_probs.item()
                    rounded_predictions_inf[r][c] = target_image_pred_inf_probs.round().item()
                else:
                    predictions_inf[r][c] = np.nan


        # Create a mask for valid (non-nan) predictions
        valid_mask = ~np.isnan(predictions_inf)

        # Use the mask to calculate the mean of valid predictions
        mean_valid_inf_predictions = np.mean(predictions_inf[valid_mask])

        inflammation_ratios.append([SlideDict['test'][slidename][1] , mean_valid_inf_predictions])
        inflammation_ratios_withWSI.append([slidename,SlideDict['test'][slidename][1] , mean_valid_inf_predictions])

    print( inflammation_ratios_withWSI)
    PlottissueDistribution(tissueratio=inflammation_ratios, model_folder=model_folder_inf)


def TestOnWSISlideFolderTissue(model_folder_inf, model_folder_tissue):
    # Path to groundtrouth CSV file
    csv_file_path = 'WSI_Path_Tissue_noIntermediate.csv'
    # Initialize an empty dictionary to store your data
    SlideDict = {}

    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile)

        # Iterate over rows in the CSV
        for row in csv_reader:
            # Assuming the format "test: {'15BHE': ['inflamed'], '15CHE': ['inflamed'], ..."
            # Split the row on ':' to separate the key and the dictionary
            key, dict_str = row[0].split(':', 1)

            # Strip any leading/trailing whitespace and remove the quotation marks
            key = key.strip().replace('"', '')

            # Use ast.literal_eval to safely evaluate the dictionary string
            dict_str = dict_str.strip()
            value_dict = ast.literal_eval(dict_str)

            # Add the key-value pair to your data dictionary
            SlideDict[key] = value_dict

    print(SlideDict)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    # get model and parameters
    trained_tissue_model, model_results, dict_hyperparameters, summary = get_model(model_folder_tissue)


    manual_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    trained_tissue_model.to(device)

    # Turn on model evaluation mode and inference mode
    trained_tissue_model.eval()

    base_path = "/mnt/thempel/scans/"

    #array for predictions
    tissue_ratios = []
    tissue_ratios_withWSI = []
    #list for path building

    for slidename in SlideDict['test']:
        # open slide with openslide

        path = find_slide_path(base_path, slidename + ".mrxs")
        print(path)
        slide = openslide.open_slide(path)

        tiles = deepzoom.DeepZoomGenerator(slide, tile_size=224,  limit_bounds=True)

        col, rows = tiles.level_tiles[14]
        predictions_inf = np.empty([rows, col])
        rounded_predictions_inf = np.empty([rows, col])


        for c in range(col):
            for r in range(rows):
                single_tile = tiles.get_tile(14, (c, r))

                # Sample usage
                np_tile = np.array(single_tile)

                # Pink lower bound in HSV
                lower_bound = np.array([135, 40, 40])
                # Purple upper bound in HSV
                upper_bound = np.array([172, 255, 255])

                if is_tile_of_interest(np_tile, lower_bound, upper_bound):
                    with torch.inference_mode():
                        # Add an extra dimension to the image
                        single_tile = manual_transforms(single_tile).unsqueeze(dim=0)

                        # Divide the image pixel values by 255 to get them between [0, 1]
                        # target_image = single_tile / 255


                        # Make a prediction on image with an extra dimension
                        target_image_pred_inf = trained_tissue_model(single_tile.cuda())


                        target_image_pred_inf_probs = torch.sigmoid(target_image_pred_inf)



                    predictions_inf[r][c] = target_image_pred_inf_probs.item()
                    rounded_predictions_inf[r][c] = target_image_pred_inf_probs.round().item()
                else:
                    predictions_inf[r][c] = np.nan


        # Create a mask for valid (non-nan) predictions
        valid_mask = ~np.isnan(predictions_inf)

        # Use the mask to calculate the mean of valid predictions
        mean_valid_inf_predictions = np.mean(predictions_inf[valid_mask])

        tissue_ratios.append([SlideDict['test'][slidename] , mean_valid_inf_predictions])
        tissue_ratios_withWSI.append([slidename,SlideDict['test'][slidename], mean_valid_inf_predictions])

    print( tissue_ratios_withWSI)
    PlottissueDistribution(tissueratio=tissue_ratios,model_folder=model_folder_tissue)
def find_slide_path(base_path, slide_name):
    for root, dirs, files in os.walk(base_path):
        if slide_name in files:
            return os.path.join(root, slide_name)
    return None
def TestOnWSISlideFolderMajorityVote(model_folder_inf, model_folder_tissue, testfolder):
    # Path to groundtrouth CSV file
    csv_file_path = 'WSI_Path_Inflamed.csv'
    # Initialize an empty dictionary to store your data
    SlideDict = {}

    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile)

        # Iterate over rows in the CSV
        for row in csv_reader:
            # Assuming the format "test: {'15BHE': ['inflamed'], '15CHE': ['inflamed'], ..."
            # Split the row on ':' to separate the key and the dictionary
            key, dict_str = row[0].split(':', 1)

            # Strip any leading/trailing whitespace and remove the quotation marks
            key = key.strip().replace('"', '')

            # Use ast.literal_eval to safely evaluate the dictionary string
            dict_str = dict_str.strip()
            value_dict = ast.literal_eval(dict_str)

            # Add the key-value pair to your data dictionary
            SlideDict[key] = value_dict

    print(SlideDict)

    class_names = cfg_hp["class_names"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # get model and parameters
    trained_inf_model, model_results, dict_hyperparameters, summary = get_model(model_folder_inf)


    manual_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    trained_inf_model.to(device)

    # Turn on model evaluation mode and inference mode
    trained_inf_model.eval()


    #array for predictions
    inflammation_ratios = []

    #list for path building
    infNoninf = {'inflamed': 'inflamed',
                 'noninflamed': 'non-inflamed'}

    for slidename in SlideDict['test']:
        # open slide with openslide

        path = "/mnt/thempel/scans/" + str(infNoninf[SlideDict['test'][slidename][0]]) + "/" +str(slidename) + ".mrxs"
        slide = openslide.open_slide(path)

        tiles = deepzoom.DeepZoomGenerator(slide, tile_size=224, overlap=112, limit_bounds=False)

        col, rows = tiles.level_tiles[15]
        predictions_inf = np.empty([rows, col])
        rounded_predictions_inf = np.empty([rows, col])


        for c in range(col):
            for r in range(rows):
                single_tile = tiles.get_tile(15, (c, r))

                # Sample usage
                np_tile = np.array(single_tile)

                # Pink lower bound in HSV
                lower_bound = np.array([135, 40, 40])
                # Purple upper bound in HSV
                upper_bound = np.array([172, 255, 255])

                if is_tile_of_interest(np_tile, lower_bound, upper_bound):
                    with torch.inference_mode():
                        # Add an extra dimension to the image
                        single_tile = manual_transforms(single_tile).unsqueeze(dim=0)

                        # Divide the image pixel values by 255 to get them between [0, 1]
                        # target_image = single_tile / 255


                        # Make a prediction on image with an extra dimension
                        target_image_pred_inf = trained_inf_model(single_tile.cuda())


                    target_image_pred_inf_probs = torch.sigmoid(target_image_pred_inf)



                    predictions_inf[r][c] = target_image_pred_inf_probs.item()
                    rounded_predictions_inf[r][c] = target_image_pred_inf_probs.round().item()
                else:
                    predictions_inf[r][c] = -1


        #Evaluation of Inflamed Tiles
        # Filter out -1 values and then count zeros and ones
        valid_inf_predictions = predictions_inf[predictions_inf != -1]



        # Filter out -1 values and then count zeros and ones

        # Count inflamed/non inflamed
        count_0 = np.count_nonzero(rounded_predictions_inf == 0)
        count_1 = np.count_nonzero(rounded_predictions_inf == 1)

        if count_0 > count_1:
            inflammation_ratios.append([slidename,SlideDict['test'][slidename][0],'noninflamed'])
        elif count_1 >= count_0:
            inflammation_ratios.append([slidename, SlideDict['test'][slidename][0] ,'inflamed'])
    print(inflammation_ratios)
    #PlotInflammedDistribution(inflammationratio=inflammation_ratios, model_folder=model_folder_inf)
def TestOnSlides(model_folder_inf: str, model_folder_tissue: str):
    # load classes
    class_names = cfg_hp["class_names"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # get model and parameters
    trained_inf_model, model_results, dict_hyperparameters, summary = get_model(model_folder_inf)
    trained_tissue_model, model_results, dict_hyperparameters, summary = get_model(model_folder_tissue)

    manual_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    trained_inf_model.to(device)
    trained_tissue_model.to(device)
    # Turn on model evaluation mode and inference mode
    trained_inf_model.eval()
    trained_tissue_model.eval()

    #array for predictions
    inflammation_ratios = []
    tissue_ratios = []


    TestSlides =   {
        '25HE': ['non-inflamed',['corpus']],
        '15BHE': ['inflamed',['antrum','corpus']],
        '36CHE': ['inflamed',['antrum','corpus']],
        '6HE': ['non-inflamed',['corpus']],
        '2CHE': ['inflamed',['antrum','corpus']],
        '77HE': ['non-inflamed',['corpus']],
        '3CHE': ['inflamed',['antrum','corpus']],
        '40BHE': ['inflamed',['antrum','corpus']],
        '15CHE': ['inflamed',['antrum']],
        '29HE': ['non-inflamed',['corpus']],
        '51HE': ['non-inflamed',['corpus']],
        '35HE': ['non-inflamed',['corpus','intermediate']],
        '66HE': ['non-inflamed',['antrum','corpus']],
        '47BHE': ['inflamed',['antrum','corpus']],
        '19CHE': ['inflamed',['antrum','corpus']],
        '20BHE': ['inflamed',['antrum']],
        #'18HE': ['non-inflamed',['corpus','intermediate']],
        #'22BHE': ['inflamed',['corpus']],
        #'23CHE': ['inflamed',['antrum','corpus']]
    }
    for slidename in TestSlides:

        # open slide with openslide
        path = "/mnt/thempel/scans/" + str(TestSlides[slidename][0]) + "/" +str(slidename) + ".mrxs"
        slide = openslide.open_slide(path)

        tiles = deepzoom.DeepZoomGenerator(slide, tile_size=224, overlap=112, limit_bounds=False)

        col, rows = tiles.level_tiles[15]
        predictions_inf = np.empty([rows, col])
        predictions_tissue= np.empty([rows, col])

        for c in range(col):
            for r in range(rows):
                single_tile = tiles.get_tile(15, (c, r))

                # Sample usage
                np_tile = np.array(single_tile)

                # Pink lower bound in HSV
                lower_bound = np.array([135, 40, 40])
                # Purple upper bound in HSV
                upper_bound = np.array([172, 255, 255])

                if is_tile_of_interest(np_tile, lower_bound, upper_bound):
                    with torch.inference_mode():
                        # Add an extra dimension to the image
                        single_tile = manual_transforms(single_tile).unsqueeze(dim=0)

                        # Divide the image pixel values by 255 to get them between [0, 1]
                        # target_image = single_tile / 255


                        # Make a prediction on image with an extra dimension
                        target_image_pred_inf = trained_inf_model(single_tile.cuda())
                        target_image_pred_tissue =  trained_tissue_model(single_tile.cuda())

                    target_image_pred_inf_probs = torch.sigmoid(target_image_pred_inf)
                    target_image_pred_tissue_probs = torch.sigmoid(target_image_pred_tissue)

                    predictions_inf[r][c] = target_image_pred_inf_probs.tolist()[0][0]
                    predictions_tissue[r][c] =target_image_pred_tissue_probs.tolist()[0][0]
                else:
                    predictions_inf[r][c] = -1
                    predictions_tissue[r][c] = -1

        #Evaluation of Inflamed Tiles
        # Filter out -1 values and then count zeros and ones
        valid_inf_predictions = predictions_inf[predictions_inf != -1]
        valid_tissue_predictions = predictions_inf[predictions_tissue != -1]

        inflammation_ratios.append([TestSlides[slidename][0] , np.mean(valid_inf_predictions)])
        tissue_ratios.append([TestSlides[slidename][1] , np.mean(valid_tissue_predictions)])

        # Filter out -1 values and then count zeros and ones




    print(inflammation_ratios)
    PlotInflammedDistribution(inflammationratio=inflammation_ratios, model_folder=model_folder_inf)

    print( tissue_ratios)
    PlottissueDistribution(tissueratio= tissue_ratios, model_folder=model_folder_tissue)
    #Show distribution with classes

def PlotInflammedDistribution(inflammationratio: np.array, model_folder):
    # Separate the probabilities
    inflamed_probs = [prob for label, prob in inflammationratio if label == 'inflamed']
    non_inflamed_probs = [prob for label, prob in inflammationratio if label == 'noninflamed']

    plt.rcParams.update({'font.size': 14})
    # Adjusting the histogram plot for better readability and specific axis scales

    plt.figure(figsize=(12, 8))

    # Creating histograms with specific bin ranges for better readability
    bins = np.arange(0, 1.1, 0.1)  # Bins from 0 to 1 with 0.1 steps
    plt.hist(inflamed_probs, bins=bins, alpha=0.7, label='Inflamed', color='red', edgecolor='black')
    plt.hist(non_inflamed_probs, bins=bins, alpha=0.7, label='Non-Inflamed', color='blue', edgecolor='black')

    plt.title('Distribution of Inflamed to Non-Inflamed ratio')
    plt.xlabel('Ratio')
    plt.xticks(bins)  # Setting x-axis ticks for each bin
    plt.yticks(range(0, int(max(plt.yticks()[0])+2)))  # Setting y-axis ticks to integers
    plt.ylabel('Frequency')

    plt.savefig(model_folder + '/WSI_Inflammation_Ratio_Distribution.png' )

    plt.legend()
    plt.grid(axis='y')  # Adding horizontal grid lines for better readability
    plt.tight_layout()  # Adjust layout
    plt.show()
def PlottissueDistribution(tissueratio: np.array, model_folder):
    # Separate the probabilities
    antrum_probs = [prob for label, prob in tissueratio if label == ['antrum']]
    corpus_probs = [prob for label, prob in tissueratio if label == ['corpus']]
    bothtissue_probs = [prob for label, prob in tissueratio if label == ['antrum','corpus']]

    plt.rcParams.update({'font.size': 14})
    # Adjusting the histogram plot for better readability and specific axis scales

    plt.figure(figsize=(12, 8))

    # Creating histograms with specific bin ranges for better readability
    bins = np.arange(0, 1.1, 0.1)  # Bins from 0 to 1 with 0.1 steps
    plt.hist(antrum_probs, bins=bins, alpha=0.7, label='Antrum', color='red', edgecolor='black')
    plt.hist(corpus_probs, bins=bins, alpha=0.7, label='Corpus', color='blue', edgecolor='black')
    plt.hist(bothtissue_probs, bins=bins, alpha=0.7, label='Antrum&Corpus', color='orange', edgecolor='black')

    plt.title('Distribution of Intermediate ratio')
    plt.xlabel('Ratio')
    plt.xticks(bins)  # Setting x-axis ticks for each bin
    plt.yticks(range(0, int(max(plt.yticks()[0])+2)))  # Setting y-axis ticks to integers
    plt.ylabel('Frequency')

    plt.savefig(model_folder + '/WSI_Tissue_Ratio_Distribution.png' )

    plt.legend()
    plt.grid(axis='y')  # Adding horizontal grid lines for better readability
    plt.tight_layout()  # Adjust layout
    plt.show()

def SlideHeatmap(heatmap_data: np.array):
    # Create a heatmap using matplotlib
    plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
    # Add colorbar to indicate values
    plt.colorbar()

    # Show the plot
    plt.show()


def is_tile_of_interest(tile, lower_thresh, upper_thresh, percentage_thresh=0.05):
    """
    Determines if a tile is of interest based on color.

    :param tile: Input tile image
    :param lower_thresh: Lower bound for HSV values
    :param upper_thresh: Upper bound for HSV values
    :param percentage_thresh: Threshold for percentage of pixels of interest
    :return: True if the tile is of interest, False otherwise
    """

    # Convert the tile to HSV color space
    hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)

    # Create a mask using inRange function
    mask = cv2.inRange(hsv, lower_thresh, upper_thresh)

    # Compute the percentage of pixels of interest
    percentage = np.sum(mask > 0) / float(mask.size)

    return percentage > percentage_thresh


# %%
